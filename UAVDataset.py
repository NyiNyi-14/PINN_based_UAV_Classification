# %% Import libraries
import torch
from torch.utils.data import Dataset
import numpy as np

# %% UAV Dataset
class UAVDataset(Dataset):
    def __init__(self, state, d_state, labels, num_classes=3):
        self.X_all = []
        self.Y_all = []

        for x_traj, dx_traj, label in zip(state, d_state, labels):
            label_onehot = np.eye(num_classes)[label]
            label_seq = np.tile(label_onehot, (len(x_traj), 1))  
            input_seq = np.hstack([x_traj, label_seq])
            
            self.X_all.append(input_seq)
            self.Y_all.append(dx_traj)

        self.X_all = np.vstack(self.X_all).astype(np.float32)
        self.Y_all = np.vstack(self.Y_all).astype(np.float32)

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.Y_all[idx]

# %%
