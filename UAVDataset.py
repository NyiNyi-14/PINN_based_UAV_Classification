# %%
import torch
from torch.utils.data import Dataset
import numpy as np

# %%
class UAVDataset(Dataset):
    def __init__(self, state, u, d_state, labels, num_classes=3):
        """
        state: list of [T state_dim] arrays
        u: list of [T input_dim] arrays
        d_state: list of [T dstate_dim] arrays
        labels: list of integers (0=quad, 1=fixed, 2=heli)
        """
        self.X_all = []
        self.Y_all = []

        for x_traj, u_traj, dx_traj, label in zip(state, u, d_state, labels):
            label_onehot = np.eye(num_classes)[label]
            label_seq = np.tile(label_onehot, (len(x_traj), 1))  # Repeat over time
            input_seq = np.hstack([x_traj, u_traj, label_seq])
            
            self.X_all.append(input_seq)
            self.Y_all.append(dx_traj)

        self.X_all = np.vstack(self.X_all).astype(np.float32)
        self.Y_all = np.vstack(self.Y_all).astype(np.float32)

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.Y_all[idx]

# %%

# x_list: list of N_traj arrays, each of shape (T, 12)
# u_list: list of N_traj arrays, each of shape (T, 4)
# dx_list: list of N_traj arrays, each of shape (T, 12)
# labels: list of N_traj integers: [0, 0, ..., 1, 1, ..., 2, 2, ...]


