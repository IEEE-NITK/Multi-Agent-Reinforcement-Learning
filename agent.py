# Agents for the Switch Riddle Game

import numpy as np 
import torch
from torch import optim
import copy


class Agent:
	def __init__(self, config_options, model, target, agent_id):
		# Pass configuration parameters using the config_options dictionary
		self.opt = config_options

		# Link model to the agent
		self.model = model
		
		self.model_target = target

		# Means to identify an agent 
		self.id = agent_id

		self.episodes_seen = 0

		self.optimizer = # Assign Adam/RMSprop to be used in learn_from_episode


		def select_action_and_comm(self, step, q, eps=0, target=False, train_mode = False):



		def calc_episode_loss(self, episode ):


		def learn_from_episode(self, episode):