import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.init_weight import init_weight

class MoENet(torch.nn.Module):
    def __init__(self, in_planes, n_expert=5):
        super(MoENet, self).__init__()
        self.tail = torch.nn.Linear(in_planes, n_expert, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.tail(x)
        output = F.log_softmax(x, dim=1)
        return output
