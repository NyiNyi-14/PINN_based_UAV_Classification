# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# %%
class PINNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# %% Example usage
model = PINNClassifier(input_dim=19, output_dim=12)  # adjust dims

# %%
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, Y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.6f}")
