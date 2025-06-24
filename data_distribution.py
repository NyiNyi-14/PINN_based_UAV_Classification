# %% NNNNNNNNNNNNNN Import Libraries NNNNNNNNNNNNNN
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %% NNNNNNNNNNNNNN Load Trajectories NNNNNNNNNNNNNN
quad_data = np.load("Quadcopter/quad_dataset.npz")
fw_data = np.load("Fixed_wings/fw_dataset.npz")
heli_data = np.load("Helicopter/heli_dataset.npz")

x_quad = quad_data["x"]
u_quad = quad_data["u"]
dx_quad = quad_data["dx"]
label_quad = quad_data["label"]

x_fw = fw_data["x"]
u_fw = fw_data["u"]
dx_fw = fw_data["dx"]
label_fw = fw_data["label"]

x_heli = heli_data["x"]
u_heli = heli_data["u"]
dx_heli = heli_data["dx"]
label_heli = heli_data["label"]

x = np.concatenate([x_quad, x_fw, x_heli], axis = 0)
u = np.concatenate([u_quad, u_fw, u_heli], axis = 0)
dx = np.concatenate([dx_quad, dx_fw, dx_heli], axis = 0)
label = np.concatenate([label_quad, label_fw, label_heli], axis = 0)

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

x_mean, x_std = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
u_mean, u_std = u.mean(axis=(0, 1)), u.std(axis=(0, 1))

x_norm = normalize(x, x_mean, x_std)
u_norm = normalize(u, u_mean, u_std)

# %% NNNNNNNNNNNNNN PCA & t-SNE NNNNNNNNNNNNNN
x_reduced = x_norm.mean(axis=1)
u_reduced = u_norm.mean(axis=1)
features = np.concatenate([x_reduced, u_reduced], axis=1)
UAV_LABELS = ['Quadcopter', 'Fixed-wing', 'Helicopter']
colors = ['red', 'blue', 'green']
component = 2

pca = PCA(n_components = component)
pca_feat = pca.fit_transform(features) 

tsne = TSNE(n_components = component, perplexity = 30, learning_rate = 200, random_state = 42)
tsne_feat = tsne.fit_transform(features)

plt.figure(figsize=(8,6))
for i in range(3):
    mask = label == i
    plt.scatter(pca_feat[mask, 0], pca_feat[mask, 1], 
                label=UAV_LABELS[i], alpha=0.6, color=colors[i])

plt.title("PCA of UAV Trajectories (Averaged Features)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(3):
    mask = label == i
    plt.scatter(tsne_feat[mask, 0], tsne_feat[mask, 1], c=colors[i], label=UAV_LABELS[i], alpha=0.6)

plt.title("t-SNE of UAV Trajectories (Averaged Features)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%



