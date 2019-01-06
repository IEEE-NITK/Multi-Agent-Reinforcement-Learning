import numpy as np 
import torch


class Arena:
	def __init__(self, opt):
		self.opt = opt


	def create_episode(self):


	def track_step_record(self):


	def run_episode(self, agents, is_training = False):



	def calc_average_reward(self, episode):
		reward = episode.r.sum()/(self.opt.bs * self.opt.num_agents)

	def train(self, agents, reset_agents = True):
		opt = self.opt

		if reset_agents:
			for agent in agents[1:]:
				agent.reset()

		for ep in range(opt.num_episodes):

			episode = self.run_episode(agents, True)
			avg_reward = self.calc_average_reward(episode)

			print("Training Epoch: ",e, "Avg Steps: ", episode.steps.float().mean().item(), 'Avg Reward: ',avg_reward)

			# Insert code from Agent class to invoke learn_from_episode here


			if e%opt.step_value == 0:
				episode = self.run_episode(agents, False)
				avg_reward = self.calc_average_reward(episode)
				rewards.append(avg_reward)

				print("Testing Epoch: ",e, "Avg Steps: ", episode.steps.float().mean().item(), 'Avg Reward: ',avg_reward)



