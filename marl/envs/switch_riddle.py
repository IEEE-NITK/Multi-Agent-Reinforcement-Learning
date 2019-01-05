import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np 

class SwitchRiddle(gym.Env):
    def __init__(self, opts):
        opts_game = {
            'game_action_space': 2,
            'game_reward_shift': 0,
            'game_comm_bits': 0,
            'game_comm_sigma': 2
        }
        opts['nsteps'] = 4 * opts['game_nagents'] - 6
        
        for key, val in opts_game.items():
            if not opts[key]:
                opts[key] = val
        self.opts = opts
        self.reward_all_live = 1 + self.opts['game_reward_shift']
        self.reward_all_die = -1 + self.opts['game_reward_shift']

        self.reset()

    
    def step(self, action):
        self.step_counter += 1
        return self.__get_reward(action)
    
    def __get_reward(self, a_t):
        for b in range(self.opts['bs']):
            active_agent = self.active_agent[b][self.step_counter]
            if a_t[b][active_agent] == 2 and self.terminal['b'] == 0:
                has_been = np.squeeze(np.sum(self.has_been[b, 1:self.step_counter, :], axis=2), axis=2)
                has_been = np.sum(np.greater(has_been, np.zeros_like(has_been), dtype=np.int16))
                if has_been == self.opts['game_nagents']:
                    self.rewards[b] = self.reward_all_live
                else:
                    self.rewards[b] = self.reward_all_die

            elif self.step_counter == self.opts['nsteps'] and self.terminal[b] == 0:
                self.terminal[b] = 1
            
        return np.copy(self.rewards), np.copy(self.terminal)

    def __get_state(self):
        state = {}
        for agent in range(1, self.opts['game_nagents']):
            state[agent] = np.empty(self.opts['bs'])

            for b in range(1, self.opts['bs']):
                state[agent][b] = 1 if self.active_agent[b][self.step_counter] == agent else 2
        
        return state
    
    def __get_action_range(self, step, agent):
        action_range = {}
        if self.opts['model_dial'] == 1:
            bound = self.opts['game_action_space']

            for i in range(self.opts['bs']):
                if self.active_agent[i][step] == agent:
                    action_range[i] = (i, (1, bound))
                else:
                    action_range[i] = (i, (1))
            return action_range
        else:
            comm_range = {}

            for i in range(self.opts['bs']):
                if self.active_agent[i][step] == agent:
                    action_range[i] = (i, (1, self.opts['game_action_space']))
                    comm_range[i] = (i, (self.opts['game_action_space'] + 1, self.opts['game_action_space_total']))
                else:
                    action_range[i] = (i, (1))
                    comm_range[i] = (i, (0, 0))
            return action_range, comm_range

    def __get_comm_limited(self, step, i):
        if self.opts['game_comm_limited']:
            action_range = {}
            for b in range(self.opts['bs']):
                if step > 1 and i == self.active_agent[b][step]:
                    action_range[i] = (self.active_agent[b][step - 1], ())
                else:
                    action_range[i] = 0

            return action_range

    def reset(self):
        self.rewards = np.zeros((self.opts['bs'], self.opts['game_nagents']), dtype='float32')
        self.has_been = np.zeros((self.opts['bs'], self.opts['nsteps'], self.opts['game_nagents']))
        self.terminal = np.zeros((self.opts['bs']))

        self.step_counter = 1
        self.active_agent = np.zeros(self.opts['bs'], self.opts['nsteps'])
        for b in range(self.opts['bs']):
            for step in range(1, self.opts['nsteps']):
                id = np.random.randint(0, self.opts['game_nagents'])
                self.active_agent[b, step] = id
                self.has_been[b, step, id] = 1

    def render(self):
        pass
