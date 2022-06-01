import torch.nn as nn
import torch.nn.functional as F
from utils import *

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

# bunch of linear laers connected L1 > ReLu(L1) > And so on
class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        
        ## This becomes (state_dim, 64, 64)
        ## We will make it using the zip trick >> 
        ## (state_dim,64) > (64,64)

        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class GaussianPolicyNet(nn.Module):
    def __init__(self, action_dim, actor_body):
        super(GaussianPolicyNet, self).__init__()

        self.actor_body = actor_body    # X ---> phi(X)

        ## The weights are implicitely defined??????
        
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs):
        
        x_phi = actor_body(obs)    ## Out =  # features dimensions
        mean_X_W = fc_action(x_phi)    ## Action dimension   >> This is avtually the mean?

        normal = torch.distributions.Normal(mean_X_W, self.std)
        
        action = normal.sample()
        log_prob = normal.log_prob(action)
        
        # entropy =  
        
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean}

class CategoricalPolicyNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body):
        super(CategoricalPolicyNet, self).__init__()
        self.actor_body = actor_body
        self.fc_action = nn.Linear(actor_body.feature_dim, action_dim)
        self.to(Config.DEVICE)

    def forward(self, obs):
        TO_DO = None
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy}

class GaussianActorCriticNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body,
                 critic_body):
        super(GaussianActorCriticNet, self).__init__()

        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.actor_params.append(self.std)
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())

        self.to(Config.DEVICE)

    def forward(self, obs):
        
        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'mean': mean,
                'v': v}

class CategoricalActorCriticNet(nn.Module):
    def __init__(self,
                 action_dim,
                 actor_body,
                 critic_body):
        super(CategoricalActorCriticNet, self).__init__()

        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs):

        return {'action': action,
                'log_pi_a': log_prob,
                'entropy': entropy,
                'v': v}

