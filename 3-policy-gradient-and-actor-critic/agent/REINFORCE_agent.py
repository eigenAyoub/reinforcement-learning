from component import *
from .BaseAgent import *

from scipy.linalg import toeplitz
import numpy as np 

class REINFORCEAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()

        # CategoricalPolicyNet // GaussianPolicyNet
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.state = self.task.reset()

    # this is training, this is run 
    def step(self):
        config = self.config
        storage = Storage(config.episode_length)  ## New one each episode
        state = self.state

        # One episode
        for _ in range(config.episode_length):
            prediction = self.network(state)          # We got {'action': action, 'log_pi_a': log_prob, 'entropy': entropy, 'mean': mean}
            next_state, reward, terminal, info = self.task.step(to_np(prediction['action']))
            self.record_online_return(info)
            storage.feed(prediction)
            storage.feed({'reward': tensor(reward).unsqueeze(-1),
                         'mask': tensor(1 - terminal).unsqueeze(-1)})
            state = next_state
            self.total_steps += 1
            if terminal:
                break
        
        ## Computing Q(x,a) >> Q(x_t,a_t)= \ sum_{t'=t}^{T-1} \gamma^{t'-t} r(x_{t'},a_{t'})
        ## Computing the loss >> policy_loss = - 1/N \sum_{t=1}^{N} log\pi_{theta}(a_t|x_t)Q(x_t, a_t)

        r = storage.extract(['reward']).reward   
        episode_length = r.shape[0]

        # this took me some thinking, lol
        col = [0**i for i in range(episode_length)]
        row = [gamma**i for i in range(episode_length)]
        toeplitz_mask = torch.tensor(toeplitz(col, row)).float()

        # [] Check how toeplitz mask is implemented in scipy. 
        rett = torch.matmul(toeplitz_mask, r)

        storage.feed({'rett': rett })

        # loss   
        entries = storage.extract(['log_pi_a', 'ret'])
        policy_loss =   -(entries.log_pi_a @ rett)/episode_length
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


        with open(self.log_dir + 'policy_loss'+str(config.seed)+'.txt', 'a') as file:
            file.write(str(policy_loss.item()) + '\n')