# %% Import Libraries 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %% Load Trajectories 
quad_data = np.load("...")
fw_data = np.load("...")
heli_data = np.load("...")

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

# %%  PCA & t-SNE 
x_reduced = x_norm.mean(axis=1)
features = np.concatenate([x_reduced], axis=1)
UAV_LABELS = ['Quadcopter', 'Fixed-wing', 'Helicopter']
colors = ['blue', 'red', 'cyan']
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

plt.xlabel(r"$\mathrm{PC \ 1}$", fontsize = 20)
plt.ylabel(r"$\mathrm{PC \ 2}$", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.tick_params(axis='both', labelsize=20) 
# plt.show()
plt.tight_layout()
# plt.savefig('.../PCA.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(8, 6))
for i in range(3):
    mask = label == i
    plt.scatter(tsne_feat[mask, 0], tsne_feat[mask, 1], c=colors[i], label=UAV_LABELS[i], alpha=0.6)

plt.xlabel(r"$\mathrm{t-SNE \ Dim \ 1}$", fontsize = 20)
plt.ylabel(r"$\mathrm{t-SNE \ Dim \ 2}$", fontsize = 20)
plt.legend(fontsize = 20)
plt.grid(True)
plt.tick_params(axis='both', labelsize=20) 
plt.tight_layout()
# plt.savefig('.../tSNE.pdf', format='pdf', bbox_inches='tight')

# %%



