# Credit to Stanford CS 234 Winter 2021 team for part of this code:
# Guillaume Genthial and Shuhui Qu, Haojun Li and Garrett Thomas

import numpy as np
from utils.test_env import EnvTest


import gym

class LinearSchedule(object):
    def __init__(self, val_begin, val_end, nsteps):
        """
        Args:
            val_begin: initial value
            val_end: end value
            nsteps: number of steps between the two values
        """
        self.curr_val = val_begin

        self.val_begin = val_begin
        self.val_end = val_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates self.curr_val

        Args:
            t: int
                frame number
        """

        if t <= self.nsteps:
            self.curr_val =  self.val_begin + t * (self.val_end - self.val_begin)/self.nsteps


class ExplorationSchedule(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(ExplorationSchedule, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, q_vals):
        """
        Returns a random action with prob curr_val, otherwise returns
        the best_action

        Args:
            q_vals: list or numpy array
                Q values for all actions

        Returns:
            an action

        """
        if np.random.uniform() < self.curr_val:
          return np.random.randint(0, np.size(q_vals))
        else:
          return np.argmax(q_vals)


if __name__=='__main__':
    # eps =  LinearSchedule(1,0)
    env = gym.make('CartPole-v0')
    exp = ExplorationSchedule(env, 1, 0, 20)




    print("hey")