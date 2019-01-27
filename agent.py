# Agents for the Switch Riddle Game

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


class CNet(nn.Module):
  def __init__(self, opts):
    super(CNet, self).__init__()
    self.opts = opts
    self.comm_size = opts['game_comm_bits']
    self.init_param_range = (-0.08, 0.08)

    ## Lookup tables for the state, action and previous action.
    self.action_lookup = nn.Embedding(opts['game_nagents'], opts['rnn_size'])
    self.state_lookup = nn.Embedding(2, opts['rnn_size'])
    self.prev_action_lookup = nn.Embedding(
        opts['game_action_space_total'], opts['rnn_size'])

    # Single layer MLP(with batch normalization for improved performance) for producing embeddings for messages.
    self.message = nn.Sequential(
        nn.BatchNorm1d(self.comm_size),
        nn.Linear(self.comm_size, opts['rnn_size']),
        nn.ReLU(inplace=True)
    )

    # RNN to approximate the agent’s action-observation history.
    self.rnn = nn.GRU(
        input_size=opts['rnn_size'], hidden_size=opts['rnn_size'], num_layers=2, batch_first=True)

    # 2 layer MLP with batch normalization, for producing output from RNN top layer.
    self.output = nn.Sequential(
        nn.Linear(opts['rnn_size'], opts['rnn_size']),
        nn.BatchNorm1d(opts['rnn_size']),
        nn.ReLU(),
        nn.Linear(opts['rnn_size'], opts['game_action_space_total'])
    )

  def get_params(self):
    return list(self.parameters())

  def reset_parameters(self):
    """
    Reset all model parameters
    """
    self.rnn.reset_parameters()
    self.action_lookup.reset_parameters()
    self.state_lookup.reset_parameters()
    self.prev_action_lookup.reset_parameters()
    self.message.apply(weight_reset)
    self.output.apply(weight_reset)
    for p in self.rnn.parameters():
      p.data.uniform_(*self.init_param_range)

  def forward(self, state, messages, hidden, prev_action, agent):
    state = Variable(torch.LongTensor(state))
    hidden = Variable(torch.FloatTensor(hidden))
    prev_action = Variable(torch.LongTensor(prev_action))
    agent = Variable(torch.LongTensor(agent))

    # Produce embeddings for rnn from input parameters
    z_a = self.action_lookup(agent)
    z_o = self.state_lookup(state)
    z_u = self.prev_action_lookup(prev_action)
    z_m = self.message(messages.view(-1, self.comm_size))

    # Add the input embeddings to calculate final RNN input.
    z = z_a + z_o + z_u + z_m
    z = z.unsqueeze(1)

    rnn_out, h = self.rnn(z, hidden)
    # Produce final CNet output q-values from GRU output.
    out = self.output(rnn_out[:, -1, :].squeeze())

    return h, out


class DRU:
	def __init__(self, sigma, comm_narrow=True, hard=False):
		self.sigma = sigma
		self.comm_narrow = comm_narrow
		self.hard = hard

	def regularize(self, m):
		m_reg = m + torch.randn(m.size()) * self.sigma
		if self.comm_narrow:
			m_reg = torch.sigmoid(m_reg)
		else:
			m_reg = torch.softmax(m_reg, 0)
		return m_reg

	def discretize(self, m):
		if self.hard:
			if self.comm_narrow:
				return (m.gt(0.5).float() - 0.5).sign().float()
			else:
				m_ = torch.zeros_like(m)
				if m.dim() == 1:
					_, idx = m.max(0)
					m_[idx] = 1.
				elif m.dim() == 2:
					_, idx = m.max(1)
					for b in range(idx.size(0)):
						m_[b, idx[b]] = 1.
				else:
					raise ValueError('Wrong message shape: {}'.format(m.size()))
				return m_
		else:
			scale = 2 * 20
			if self.comm_narrow:
				return torch.sigmoid((m.gt(0.5).float() - 0.5) * scale)
			else:
				return torch.softmax(m * scale, -1)

	def forward(self, m, train_mode):
		if train_mode:
			return self.regularize(m)
		else:
			return self.discretize(m)


class Agent:
  def __init__(self, opts, game, model, target, agent_no):
    self.game = game
    self.opts = opts
    self.model = model
    self.model_target = target
    self.id = agent_no

    # Make target model not trainable
    for param in target.parameters():
      param.requires_grad = False

    self.episodes = 0
    self.dru = DRU(opts['game_comm_sigma'])
    self.optimizer = optim.RMSprop(
        params=model.get_params(), lr=opts['lr'], momentum=opts['momentum'])

  def reset(self):
    self.model.reset_parameters()
    self.model_target.reset_parameters()
    self.episodes = 0

  def _eps_flip(self, eps):
    return np.random.rand(self.opts['bs']) < eps

  def _random_choice(self, items):
    return torch.from_numpy(np.random.choice(items, 1)).item()

  def select(self, step, q, eps=0, target=False, train=False):
    if not train:
      eps = 0  # Pick greedily during test

    opts = self.opts

    # Get the action range and communication range for the agent for the current time step.
    action_range, comm_range = self.game.get_action_range(step, self.id)

    action = torch.zeros(opts['bs'], dtype=torch.long)
    action_value = torch.zeros(opts['bs'])
    comm_vector = torch.zeros(opts['bs'], opts['game_comm_bits'])

    select_random_a = self._eps_flip(eps)
    for b in range(opts['bs']):
      q_a_range = range(0, opts['game_action_space'])
      a_range = range(action_range[b, 0].item() - 1, action_range[b, 1].item())
      if select_random_a[b]:
        # select action randomly (to explore the state space)
        action[b] = self._random_choice(a_range)
        action_value[b] = q[b, action[b]]
      else:
        action_value[b], action[b] = q[b, a_range].max(
            0)  # select action greedily
      action[b] = action[b] + 1

      q_c_range = range(opts['game_action_space'],
                        opts['game_action_space_total'])
      if comm_range[b, 1] > 0:
        # if the agent can communicate for the given time step
        c_range = range(comm_range[b, 0].item() - 1, comm_range[b, 1].item())
        # real-valued message from DRU based on q-values
        comm_vector[b] = self.dru.forward(q[b, q_c_range], train_mode=train)
    return (action, action_value), comm_vector

  def get_loss(self, episode):
    opts = self.opts
    total_loss = torch.zeros(opts['bs'])
    for b in range(opts['bs']):
      b_steps = episode.steps[b].item()
      for step in range(b_steps):
        record = episode.step_records[step]
        for i in range(opts['game_nagents']):
          td_action = 0
          r_t = record.r_t[b][i]
          q_a_t = record.q_a_t[b][i]

          # Calculate td loss for environment action
          if record.a_t[b][i].item() > 0:
            if record.terminal[b].item() > 0:
              td_action = r_t - q_a_t
            else:
              next_record = episode.step_records[step + 1]
              q_next_max = next_record.q_a_max_t[b][i]
              td_action = r_t = opts['gamma'] * q_next_max - q_a_t

          loss_t = td_action ** 2
          total_loss[b] = total_loss[b] + loss_t
    loss = total_loss.sum()
    return loss / (opts['bs'] * opts['game_nagents'])

  def update(self, episode):
    self.optimizer.zero_grad()
    loss = self.get_loss(episode)
    loss.backward()
    # Clip gradients for stable training
    clip_grad_norm_(parameters=self.model.get_params(), max_norm=10)
    self.optimizer.step()
    self.episodes += 1

    # Update target model
    if self.episodes % self.opts['step_target'] == 0:
      self.model_target.load_state_dict(self.model.state_dict())
