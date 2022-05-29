import numpy as np
import sys
import torch
import torch.nn as nn
from utils.replay_buffer import ReplayMemory, Transition
from utils.general import export_plot, CSVLogger
from collections import deque

class DQNTrainer(object):
    def __init__(self,
                 env,
                 exploration_schedule,
                 lr_schedule,
                 config,
                 q_function
                 ):

        self.env = env
        self.state_shape = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.config = config
        
        self.q_network = q_function(self.state_shape,
                                    self.num_actions)
        self.target_network = q_function(self.state_shape,
                                         self.num_actions)

        # initializing weights
        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)

        self.q_network.apply(init_weights)
        self.target_network.load_state_dict(
            self.q_network.state_dict())

        self.exploration_schedule = exploration_schedule
        self.lr_schedule = lr_schedule

        self.q_optimizer = torch.optim.RMSprop(
            self.q_network.parameters(), alpha=0.95, eps=0.01)

    def learn(self):
        '''
        :param env:
        :param q_fun:
        :param exploration_schedule:
        :param lr_schedule:
        :param q_optimizer:
        :param config:
        :return:
        '''
        replay_buffer = ReplayMemory(self.config.replay_buffer_size)
        data_dict = {
            'Timestep': 0,
            'Training Rewards': 0,
            'Max Q': 0,
            'Eval Rewards': 0,
            'Loss': 0
        }

        fieldnames = [key for key, _ in data_dict.items()]
        csv_logger = CSVLogger(fieldnames=fieldnames,
                               filename=self.config.csv_dir)
        rewards = deque(maxlen=self.config.num_episodes_eval)
        max_q_values = deque(maxlen=self.config.num_episodes_eval)
        loss_val = params_norm = float('inf')
        t = last_eval = 0

        ## 
        scores_eval = [self.evaluate()]
        
        train_rewards = []
        
        num_episodes = 0


        while t < self.config.num_timesteps:
            num_episodes += 1
            state = self.env.reset()
            done = False
            episode_reward = 0

            ## >>> ONE-EPISODE  -- Begin

            while not done and (t < self.config.num_timesteps):
                t += 1
                last_eval += 1
                # choose action according to current Q and exploration

                with torch.no_grad(): 
                    q_vals = self.q_network(torch.FloatTensor(state))
                    action = self.exploration_schedule.get_action(q_vals.numpy())
                    

                if t > self.config.learning_start:
                    ## After t - 100 > 1000, epsilon stays at 0.1
                    self.exploration_schedule.update(t - self.config.learning_start)

                max_q_values.append(max(q_vals))  # why, probb some shitty visual

                # usual stuff
                # print("yo", print(action))
                next_state, reward, done, info = self.env.step(action)

                replay_buffer.push(torch.tensor(state),
                                   torch.tensor(action).unsqueeze(0),
                                   torch.tensor(next_state),
                                   torch.tensor(reward).unsqueeze(0),
                                   torch.tensor(done, dtype=torch.bool).unsqueeze(0))
                
                state = next_state

                # TRAIN
                # (t > self.config.learning_start) and (len(replay_buffer) >= self.config.batch_size) and (t % self.config.learning_freq == 0):
                if (t > self.config.learning_start) and (len(replay_buffer) >= self.config.batch_size):
                    loss_val, params_norm = self.training_step(t, replay_buffer, self.lr_schedule.curr_val)
                    ## when the target is update? how? Everything is handled in the training_step
                    self.lr_schedule.update(t - self.config.learning_start)

                # Some useful printing, each 100 transitions.
                if (t % 100 == 0) and (t > 100):
                #if (t % self.config.log_freq == 0) and (t > self.config.learning_start):
                    if len(rewards) > 0:
                        print(
                            f'SGD Step {t - self.config.learning_start} | '
                            f'Episode {num_episodes} | '
                            f'E[R] {np.mean(rewards):.4f} | '
                            f'Max R {np.max(rewards):.4f} | '
                            f'Max Q {np.mean(max_q_values):.4f} | '
                            f'Params Norm {params_norm:.4f} | '
                            f'Loss {loss_val:.4f} | '
                            f'lr {self.lr_schedule.curr_val:.6f} | '
                            f'eps {self.exploration_schedule.curr_val:.3f}')
                        sys.stdout.flush()

                episode_reward += reward

                if done or t >= self.config.num_timesteps:
                    break

            ## >>> ONE-EPISODE  -- END

            rewards.append(episode_reward)

            if  last_eval >= 100 :    
            #if (t > self.config.learning_start) and (last_eval >= self.config.eval_freq):
            # 100 update after last_time
                last_eval = 0
                train_rewards += [np.mean(rewards)]
                export_plot(train_rewards, "Episode Rewards",
                            self.config.train_plot_dir)
                scores_eval += [self.evaluate()]
                data_dict = {
                    'Timestep': t,
                    'Training Rewards': float(np.mean(rewards)),
                    'Max Q': np.mean(max_q_values),
                    'Eval Rewards': float(scores_eval[-1]),
                    'Loss': float(loss_val)
                }
                csv_logger.writerow(data_dict)
                export_plot(scores_eval, "Episode Rewards",
                            self.config.plot_dir)

        print('Training Finished')

        torch.save(self.q_network.state_dict(),self.config.model_dir)
        scores_eval += [self.evaluate()]
        train_rewards += [np.mean(rewards)]
        export_plot(train_rewards, "Episode Rewards", self.config.train_plot_dir)
        export_plot(scores_eval, "Episode Rewards", self.config.plot_dir)
        csv_logger.close()

    def training_step(self, t, replay_buffer, lr):
        '''
        :param t:
        :param replay_buffer:
        :param lr:
        :return:
        '''
        
        ## batch size set to 32
        batch = zip(*replay_buffer.sample(32))

        state_batch = torch.cat(next(batch)).float().view(32, -1)
        action_batch = torch.cat(next(batch)).long().view(32, -1)
        next_state_batch = torch.cat(next(batch)).float().view(32, -1)
        reward_batch = torch.cat(next(batch)).view(32, -1)
        done_batch = torch.cat(next(batch)).view(32, -1).squeeze()

        if self.config.double == True:
            loss = self.compute_DoubleDQN_loss(state_batch,
                                               action_batch,
                                               reward_batch,
                                               next_state_batch,
                                               done_batch)
        else:
            loss = self.compute_DQN_loss(state_batch,
                                            action_batch,
                                            reward_batch,
                                            next_state_batch, 
                                            done_batch)

        self.q_optimizer.zero_grad()

        loss.backward()

        total_param_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.clip_val)

        # set optimizer learning rate
        for group in self.q_optimizer.param_groups:
            group['lr'] = lr
        

        self.q_optimizer.step()

        if t % self.config.target_update_freq == 0:
        #if t / self.config.learning_freq % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if t % self.config.saving_freq == 0:
            torch.save(self.q_network.state_dict(), self.config.model_dir)

        return loss.item(), total_param_norm


    def compute_DQN_loss(self, state_batch, action_batch,
                         reward_batch, next_state_batch, done_batch):
        '''
        :param state_batch: (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of states collected during
                training (i.e. s)
        :param action_batch: (torch LongTensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
        :param reward_batch: (torch tensor) shape = (batch_size,)
                The rewards that you actually got at each step (i.e. r)
        :param (torch tensor) shape = (batch_size x state_dims),
                The batched tensor of next states collected during
                training (i.e. s')
        :param done_batch: (torch tensor) shape = (batch_size,)
                A boolean mask of examples where we reached the terminal state
        :return: loss: (torch tensor) shape = (1)
        '''
        with torch.no_grad():
            max_a, _  = torch.max(self.target_network(next_state_batch), dim = 1)
            max_a[done_batch==True] = 0

        loss_vector = (reward_batch + self.config.gamma*max_a.view(32,1)   -  self.q_network(state_batch).gather(1, action_batch))**2
        loss = torch.mean(loss_vector)
        
        return loss


    def compute_DoubleDQN_loss(self, state_batch, action_batch,
                               reward_batch, next_state_batch,
                               done_batch):
        '''
        :param state_batch: Tensor (batch_size x state_dims),
        batched tensor of states collected during training
        :param action_batch: LongTensor (batch_size x action_dims)
        :param reward_batch: Tensor (batch_size x 1)
        :param next_state_batch: Tensor (batch_size x state_dims)
        batched tensor of next states
        :param done_batch: Tensor of bools (batch_size x 1)
        :return: scalar
        '''
        '''
        TODO:               
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                      = r + gamma * Q_target(s', argmax_{a'} Q(s', a'))
            loss = (Q_samp(s) - Q(s, a))^2
            
        Recall that there should not be any gradients passed 
        through the target network (Hint: you can use "with 
        torch.no_grad()")
        '''
        loss = None
        ##############################################################
        ############### YOUR CODE HERE - 8-10 lines ##################


        ##############################################################
        ######################## END YOUR CODE #######################
        return loss


    def evaluate(self):
        rewards = []
        for ep in range(self.config.num_episodes_eval):
            episode_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                state_norm = torch.FloatTensor(state / self.config.high)
                with torch.no_grad():
                    q_vals = self.q_network(state_norm)
                    action = np.argmax(q_vals.numpy())
                next_state, reward, done, info = self.env.step(action)
                next_state = None if done else next_state
                state = next_state
                episode_reward += reward
            rewards.append(episode_reward)
        avg_reward = np.mean(rewards)
        std_error = np.sqrt(np.var(rewards) / len(rewards))
        print(f'Eval average reward: {avg_reward:04.2f} +/-'
              f' {std_error:04.2f}')
        sys.stdout.flush()
        return avg_reward
