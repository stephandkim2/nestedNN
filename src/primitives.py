import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplyOp(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer: compute (x+y), (x-y)
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc1.weight.data = torch.tensor([
            [1.0,  1.0],   # (x+y)
            [1.0, -1.0]    # (x-y)
        ])
        
        # Final linear layer: combine squared terms
        self.fc2 = nn.Linear(2, 1, bias=False)
        self.fc2.weight.data = torch.tensor([[0.25, -0.25]])  # (1/4)[(x+y)^2 - (x-y)^2]

    def forward(self, x):
        # x shape: [batch, 2], where x[:,0]=x, x[:,1]=y
        h = self.fc1(x)       # [batch, 2] -> (x+y), (x-y)
        h2 = h**2             # elementwise square
        out = self.fc2(h2)    # scalar = xy
        # duplicate to make output 2D
        return torch.cat([out, out], dim=-1)  # [batch, 2]

class AddOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)
        self.fc.weight.data = torch.tensor([[1.0, 1.0]])  # x+y

    def forward(self, x):
        out = self.fc(x)
        return torch.cat([out, out], dim=-1)  # duplicate
    
class ReLUOp(nn.Module):
    def forward(self, x):
        # x: [batch, 2]
        out = torch.relu(x)   # elementwise ReLU
        return out 
    
class SoftMaxOp(nn.Module):
    def __init__(self, beta=10.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        # x: [batch, 2]
        weights = torch.softmax(self.beta * x, dim=1)    # [batch, 2]
        out = (weights * x).sum(dim=1, keepdim=True)     # weighted average
        return torch.cat([out, out], dim=-1)             # duplicate
    
class SoftModulo(nn.Module):
    def __init__(self, m=2*torch.pi):
        super().__init__()
        self.m = m

    def forward(self, x):
        # soft periodic wrap
        return x - self.m * torch.tanh(x / self.m)
    
