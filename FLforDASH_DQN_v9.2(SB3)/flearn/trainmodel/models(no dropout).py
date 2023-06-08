import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, num_states,  num_actions):
        super(Model, self).__init__()

        self.hidden1 = nn.Linear(num_states, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.output = nn.Linear(128, num_actions)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def predict(self, inputs):
        x= torch.tanh(self.hidden1(inputs))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        output = self.output(x)
        # output = torch.softmax(self.output(x))
        return output

