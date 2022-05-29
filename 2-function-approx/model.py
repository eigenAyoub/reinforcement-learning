import torch.nn as nn

class QModel(nn.Module):
    def __init__(self, in_features=128, num_actions=18):
        '''
        Architecture of Q-function
        :param in_features: dimension of input
        :param num_actions: dimension of output
        '''
        '''
        TODO: Define the architecture of the model here. 
        You may find nn.Sequential helpful.
        '''
        super(QModel, self).__init__()

        self.model = None

        self.lin1 = nn.Linear(in_features, 10)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(10, num_actions)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x
        #return self.model(x)
