import numpy as np 

class SwitchRiddle():
    def __init__(self, opts):
        opts_game = {
            'game_action_space': 2,
            'game_reward_shift': 0,
            'game_comm_bits': 0,
            'game_comm_sigma': 2
        }
        opts['nsteps'] = 4 * opts['game_nagents'] - 6
        
        for key, val in opts_game.items():
            # print(key)
            if key not in opts.keys():
                opts[key] = val
        self.opts = opts
        self.reward_all_live = 1 + self.opts['game_reward_shift']
        self.reward_all_die = -1 + self.opts['game_reward_shift']

        self.reset()

    
    def step(self, action):
        reward, terminal = self.get_reward(action)
        self.step_counter += 1

        return reward, terminal

    def get_reward(self, a_t):
        for b in range(self.opts['bs']):
            active_agent = self.active_agent[b][self.step_counter] - 1
            if a_t[b][active_agent] == 2 and self.terminal[b] == 0:
                has_been = np.squeeze(np.sum(self.has_been[b, :self.step_counter + 1, :], axis=2), axis=2)
                has_been = np.sum(np.greater(has_been, np.zeros_like(has_been), dtype=np.int16))
                if has_been == self.opts['game_nagents']:
                    self.rewards[b] = self.reward_all_live
                else:
                    self.rewards[b] = self.reward_all_die

            elif self.step_counter == self.opts['nsteps'] and self.terminal[b] == 0:
                self.terminal[b] = 1
            
        return np.copy(self.rewards), np.copy(self.terminal)

    def get_state(self):
        state = np.zeros((self.opts['bs'], self.opts['game_nagents']))
        for agent in range(1, self.opts['game_nagents'] + 1):
            for b in range(1, self.opts['bs']):
                state[b][agent - 1] = 1 if self.active_agent[b][self.step_counter] == agent else 0
        
        return state
    
    def get_action_range(self, step, agent):
        action_range = np.zeros((self.opts['bs'], 2), dtype=np.long)
        comm_range = np.zeros((self.opts['bs'], 2), dtype=np.long)
        for b in range(self.opts['bs']): 
            if self.active_agent[b][step] == agent:
                action_range[b] = np.array([1, self.opts['game_action_space']], dtype=np.long)
                comm_range[b] = np.array([self.opts['game_action_space'] + 1, self.opts['game_action_space_total']], dtype=np.long)
            else:
                action_range[b] = np.array([1, 1], dtype=np.long)
        
        return action_range, comm_range

    def get_comm_limited(self, step, agent):
        if self.opts['game_comm_limited']:
            comm_lim = np.zeros(self.opts['bs'], dtype=np.long)
            for b in range(self.opts['bs']):
                if step > 0 and agent == self.active_agent[b][step]:
                    comm_lim[agent] = (self.active_agent[b][step - 1], ())
                else:
                    comm_lim[agent] = 0

            return comm_lim

    def reset(self):
        self.rewards = np.zeros((self.opts['bs'], self.opts['game_nagents']))
        self.has_been = np.zeros((self.opts['bs'], self.opts['nsteps'], self.opts['game_nagents']))
        self.terminal = np.zeros((self.opts['bs']), dtype=np.long)

        self.step_counter = 0
        self.active_agent = np.zeros((self.opts['bs'], self.opts['nsteps']), dtype=np.long)
        for b in range(self.opts['bs']):
            for step in range(self.opts['nsteps']):
                id = 1 + np.random.randint(0, self.opts['game_nagents'])
                self.active_agent[b, step] = id
                self.has_been[b, step, id - 1] = 1

    def god_strategy_reward(self, steps):
        reward = np.zeros(self.opts['bs'])
        for b in range(self.opts['bs']):
            has_been = np.squeeze(np.sum(self.has_been[b, :self.step_counter + 1, :], axis=2), axis=2)
            has_been = np.sum(np.greater(has_been, np.zeros_like(has_been), dtype=np.int16))
            if has_been == self.opts['game_nagents']:
                reward[b] = self.reward_all_live

        return reward

    def render(self, b=0):
        print('has been:', self.has_been[b])
        print('num has been:', self.has_been[b].sum(0).gt(0).sum().item())
        print('active agents: ', self.active_agent[b])
        print('reward:', self.rewards[b])
