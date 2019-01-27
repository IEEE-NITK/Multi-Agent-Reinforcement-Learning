from utils import DotDic
import copy
import torch
import torch.nn as nn
from env.switch_riddle import SwitchGame
from agent import CNet, DRU, Agent

class Arena:
  def __init__(self, opt, game):
    self.opt = opt
    self.game = game
    self.eps = opt.eps

  def create_episode(self):
    opt = self.opt
    episode = DotDic({})
    episode.steps = torch.zeros(opt.bs).int()
    episode.ended = torch.zeros(opt.bs).int()
    episode.r = torch.zeros(opt.bs, opt.game_nagents).float()
    episode.step_records = []

    return episode

  def create_step_record(self):
    """
    Returns an empty step record to store the data from each step in the episode
    """
    opt = self.opt
    record = DotDic({})
    record.s_t = None
    record.r_t = torch.zeros(opt.bs, opt.game_nagents)
    record.terminal = torch.zeros(opt.bs)

    record.agent_inputs = []
    record.a_t = torch.zeros(opt.bs, opt.game_nagents, dtype=torch.long)
    record.comm = torch.zeros(opt.bs, opt.game_nagents, opt.game_comm_bits)
    record.comm_target = record.comm.clone()

    record.hidden = torch.zeros(opt.game_nagents, 2, opt.bs, opt.rnn_size)
    record.hidden_target = torch.zeros(
        opt.game_nagents, 2, opt.bs, opt.rnn_size)

    record.q_a_t = torch.zeros(opt.bs, opt.game_nagents)
    record.q_a_max_t = torch.zeros(opt.bs, opt.game_nagents)

    return record

  def run_episode(self, agents, train_mode=False):
    opt = self.opt
    game = self.game
    game.reset()
    self.eps = self.eps * opt.eps_decay

    step = 0
    episode = self.create_episode()
    s_t = game.get_state()
    # Intialize step record
    episode.step_records.append(self.create_step_record())
    episode.step_records[-1].s_t = s_t
    episode_steps = train_mode and opt.nsteps + 1 or opt.nsteps
    while step < episode_steps and episode.ended.sum() < opt.bs:
      # Run through the episode
      episode.step_records.append(self.create_step_record())

      for i in range(1, opt.game_nagents + 1):
        agent = agents[i]
        agent_idx = i - 1

        # Retrieve model inputs from the records
        comm = episode.step_records[step].comm.clone()
        comm_limited = self.game.get_comm_limited(step, agent.id)
        if comm_limited is not None:
          comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
          for b in range(opt.bs):
            if comm_limited[b].item() > 0:
              comm_lim[b] = comm[b][comm_limited[b] - 1]
          comm = comm_lim
        else:
          comm[:, agent_idx].zero_()
        prev_action = torch.ones(opt.bs, dtype=torch.long)
        if not opt.model_dial:
          prev_message = torch.ones(opt.bs, dtype=torch.long)
        for b in range(opt.bs):
          if step > 0 and episode.step_records[step - 1].a_t[b, agent_idx] > 0:
            prev_action[b] = episode.step_records[step - 1].a_t[b, agent_idx]
        batch_agent_index = torch.zeros(
            opt.bs, dtype=torch.long).fill_(agent_idx)

        agent_inputs = {
            'state': episode.step_records[step].s_t[:, agent_idx],
            'messages': comm,
            'hidden': episode.step_records[step].hidden[agent_idx, :],
            'prev_action': prev_action,
            'agent': batch_agent_index
        }
        episode.step_records[step].agent_inputs.append(agent_inputs)

        # Get Q-values from CNet
        hidden_t, q_t = agent.model(**agent_inputs)
        episode.step_records[step + 1].hidden[agent_idx] = hidden_t.squeeze()
        # Pick actions based on q-values
        (action, action_value), comm_vector = agent.select(
            step, q_t, eps=self.eps, train=train_mode)

        episode.step_records[step].a_t[:, agent_idx] = action
        episode.step_records[step].q_a_t[:, agent_idx] = action_value
        episode.step_records[step + 1].comm[:, agent_idx] = comm_vector

      a_t = episode.step_records[step].a_t
      episode.step_records[step].r_t, episode.step_records[step].terminal = self.game.step(
          a_t)

      # Update episode record rewards
      if step < opt.nsteps:
        for b in range(opt.bs):
          if not episode.ended[b]:
            episode.steps[b] = episode.steps[b] + 1
            episode.r[b] = episode.r[b] + episode.step_records[step].r_t[b]

          if episode.step_records[step].terminal[b]:
            episode.ended[b] = 1

      # Update target network during training
      if train_mode:
        for i in range(1, opt.game_nagents + 1):
          agent_target = agents[i]
          agent_idx = i - 1

          agent_inputs = episode.step_records[step].agent_inputs[agent_idx]
          comm_target = agent_inputs.get('messages', None)

          comm_target = episode.step_records[step].comm_target.clone()
          comm_limited = self.game.get_comm_limited(step, agent.id)
          if comm_limited is not None:
            comm_lim = torch.zeros(opt.bs, 1, opt.game_comm_bits)
            for b in range(opt.bs):
              if comm_limited[b].item() > 0:
                comm_lim[b] = comm_target[b][comm_limited[b] - 1]
            comm_target = comm_lim
          else:
            comm_target[:, agent_idx].zero_()

          agent_target_inputs = copy.copy(agent_inputs)
          agent_target_inputs['messages'] = Variable(comm_target)
          agent_target_inputs['hidden'] = episode.step_records[step].hidden_target[agent_idx, :]
          hidden_target_t, q_target_t = agent_target.model_target(
              **agent_target_inputs)
          episode.step_records[step +
                               1].hidden_target[agent_idx] = hidden_target_t.squeeze()

          (action, action_value), comm_vector = agent_target.select(
              step, q_target_t, eps=0, target=True, train=True)

          episode.step_records[step].q_a_max_t[:, agent_idx] = action_value
          episode.step_records[step +
                               1].comm_target[:, agent_idx] = comm_vector

      step = step + 1
      if episode.ended.sum().item() < opt.bs:
        episode.step_records[step].s_t = self.game.get_state()

    episode.game_stats = self.game.get_stats(episode.steps)

    return episode

  def average_reward(self, episode, normalized=True):
    reward = episode.r.sum()/(self.opt.bs * self.opt.game_nagents)
    if normalized:
      oracle_reward = episode.game_stats.oracle_reward.sum()/self.opt.bs
      if reward == oracle_reward:
        reward = 1
      elif oracle_reward == 0:
        reward = 0
      else:
        reward = reward/oracle_reward
    return float(reward)

  def train(self, agents, reset=True, verbose=False, test_callback=None):
    opt = self.opt
    if reset:
      for agent in agents[1:]:
        agent.reset()

    rewards = []
    for e in range(opt.nepisodes):
      episode = self.run_episode(agents, train_mode=True)
      norm_r = self.average_reward(episode)
      if verbose:
        print('train epoch:', e, 'avg steps:',
              episode.steps.float().mean().item(), 'avg reward:', norm_r)
      agents[1].update(episode)

      if e % opt.step_test == 0:
        episode = self.run_episode(agents, train_mode=False)
        norm_r = self.average_reward(episode)
        rewards.append(norm_r)
        print('TEST EPOCH:', e, 'avg steps:',
              episode.steps.float().mean().item(), 'avg reward:', norm_r)


def main():
  game = SwitchGame(DotDic(opts))
  cnet = CNet(opts)
  cnet_target = copy.deepcopy(cnet)
  agents = [None]
  for i in range(1, opts['game_nagents'] + 1):
    agents.append(Agent(DotDic(opts), game=game, model=cnet, target=cnet_target, agent_no=i))

arena = Arena(DotDic(opts), game)
arena.train(agents)


if __name__ == '__main__':
  main()