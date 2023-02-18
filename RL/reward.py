from torch import nn
import torch

LEARNING_RATE = 0.001
class Reward(nn.Module):
    def __init__(self, input_dim, num_actions=2):
        super(Reward, self).__init__()
      
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.functional.binary_cross_entropy
    def forward(self, inputs):
        x = self.leaky_relu(self.l1(inputs))
        x = self.leaky_relu(self.l2(x))
        return x
    
    def compute_proba(self, query1, query2):
        out1, out2 = self.forward(torch.Tensor(query1)), self.forward(torch.Tensor(query2))
        sum_p1 = torch.exp(torch.sum(out1))
        sum_p2 = torch.exp(torch.sum(out2))
        p1 = sum_p1/(sum_p1 + sum_p2)
        p2 = 1 - p1
        return p1, p2

    def train_step(self, query1, query2, human_pref):
        p1, p2 = self.compute_proba(query1, query2)
        if human_pref < 2:
          label = torch.nn.functional.one_hot(torch.tensor(human_pref), 2).float()
        else:
          label = torch.Tensor([0.5, 0.5]).float()

        self.optimizer.zero_grad()
        loss = self.criterion(torch.stack([p1, p2]), label)
        loss.backward() 
        self.optimizer.step() 
        return loss.item()
