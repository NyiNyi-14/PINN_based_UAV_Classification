# %% Import libraries
import torch.nn as nn

# %% ResNEt style PINN
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()  

    def forward(self, x):

        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1) + h1)
        h3 = self.act(self.fc3(h2) + 0.7*h2 + 0.3*h1) 
        h4 = self.act(self.fc4(h3) + h3) 
        h5 = self.act(self.fc5(h4) + 0.6*h4 + 0.4*h1)
        out = self.fc_out(0.5*h5 + 0.3*h3 + 0.2*h1)
        return out
    
# %%
