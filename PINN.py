# %%
import torch.nn as nn

# %%
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128) # hidden 1
        self.fc2 = nn.Linear(128, 128) # hidden 2
        self.fc3 = nn.Linear(128, 128) # hidden 3
        self.fc4 = nn.Linear(128, output_dim) # out
        self.act = nn.Tanh() #activation func
        
    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1) + h1) # skip con
        h3 = self.act(self.fc3(h2) + h2) #skip con
        out = self.fc4(h3)
        return out

# %%