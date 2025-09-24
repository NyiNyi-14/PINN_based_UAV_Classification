# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

os.chdir("...")
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from UAVDataset import UAVDataset
from PINN_Classifier import PINN_Classifier
from Noise import Noise

# %% Time Frame
duration = 10
dt = 0.01
time = np.arange(0, duration, dt)
classes = 3

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

# %% Data Preprocessing
x = np.concatenate([x_quad, x_fw, x_heli], axis = 0)
u = np.concatenate([u_quad, u_fw, u_heli], axis = 0)
dx = np.concatenate([dx_quad, dx_fw, dx_heli], axis = 0)
label = np.concatenate([label_quad, label_fw, label_heli], axis = 0)

x_train, x_temp, u_train, u_temp, dx_train, dx_temp, label_train, label_temp = train_test_split(
    x, u, dx, label, test_size = 0.2, random_state = 14)

x_val, x_test, u_val, u_test, dx_val, dx_test, label_val, label_test = train_test_split(
    x_temp, u_temp, dx_temp, label_temp, test_size = 0.5, random_state = 14)

x_mean, x_std = x_train.mean(axis=(0, 1)), x_train.std(axis=(0, 1))
u_mean, u_std = u_train.mean(axis=(0, 1)), u_train.std(axis=(0, 1))
dx_mean, dx_std = dx_train.mean(axis=(0, 1)), dx_train.std(axis=(0, 1))

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

x_train_norm = normalize(x_train, x_mean, x_std)
x_val_norm   = normalize(x_val, x_mean, x_std)
x_test_norm  = normalize(x_test, x_mean, x_std)

u_train_norm = normalize(u_train, u_mean, u_std)
u_val_norm   = normalize(u_val, u_mean, u_std)
u_test_norm  = normalize(u_test, u_mean, u_std)

dx_train_norm = normalize(dx_train, dx_mean, dx_std)
dx_val_norm   = normalize(dx_val, dx_mean, dx_std)
dx_test_norm  = normalize(dx_test, dx_mean, dx_std)

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

np.savez("results/normalization_stats.npz", 
         x_mean=x_mean, x_std=x_std, 
         u_mean=u_mean, u_std=u_std, 
         dx_mean=dx_mean, dx_std=dx_std)

train_dataset = UAVDataset(x_train_norm, dx_train_norm, label_train, num_classes = classes)
val_dataset   = UAVDataset(x_val_norm, dx_val_norm, label_val, num_classes = classes)
test_dataset  = UAVDataset(x_test_norm, dx_test_norm, label_test, num_classes = classes)

train_loader = DataLoader(train_dataset, batch_size = 256, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size = 1000, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size = 1000, shuffle=False)

# %% Testing DataLoader
for X_batch, Y_batch in test_loader:
    print("X_batch shape:", X_batch.shape)  
    print("Y_batch shape:", Y_batch.shape) 
    print("First input sample:\n", X_batch[0])
    print("First label (dstate):\n", Y_batch[0])
    break  

for i, (X_batch, Y_batch) in enumerate(test_loader):
    print(f"Batch {i} - X shape: {X_batch.shape}, Y shape: {Y_batch.shape}")

# %% Loading Pre-trained Model and Inference
input_dim = x.shape[-1] + classes
output_dim = dx.shape[-1]
save_dir = "results"
model_path = os.path.join(save_dir, "Pinn_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = PINN_Classifier(
    input_dim=input_dim,
    output_dim=output_dim,
    lr=1e-3,  
    delta_loss=1e-5,
    interval=30
)

try:
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    classifier.model.load_state_dict(state_dict)
    classifier.model.to(device)
    print(f"Model loaded successfully into classifier")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

classifier.model.eval()

# %% Getting subset of test data for classification
n_traj = len(test_dataset) // len(time)
selected_traj_ids = random.sample(range(n_traj), 300)

selected_indices = []
for tid in selected_traj_ids:
    start = tid * len(time)
    selected_indices.extend(range(start, start + len(time)))

from torch.utils.data import Subset
test_subset = Subset(test_dataset, selected_indices)
test_subset_loader = DataLoader(test_subset, batch_size=len(time), shuffle=False)

# %% Classification on test data
true_class = []
predictions = []
losses = []
probabilities = []

for X_batch, Y_batch in test_subset_loader:
    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
    x = X_batch[:, :12]
    labels = X_batch[:, 12:]

    pred_class, raw_losses, confidences = classifier.classify(x=x, dx=Y_batch)

    true_class.append(torch.argmax(labels, dim=1).cpu().numpy())
    predictions.append(pred_class)
    losses.append(raw_losses)
    probabilities.append(confidences)

UAV_LABELS = {
0: "Quadcopter",
1: "Fixed-wing",
2: "Helicopter"
}

true_lab_stored = np.array([UAV_LABELS[true_class[i][0]] for i in range(len(test_subset_loader))])
pred_lab_stored = np.array([UAV_LABELS[predictions[i]] for i in range(len(test_subset_loader))])
probabilities_stored = np.array([probabilities[i] for i in range(len(test_subset_loader))])

check = true_lab_stored == pred_lab_stored

table = {
    "True Label": true_lab_stored,
    "uad %": probabilities_stored[:, 0],
    "FW %": probabilities_stored[:, 1],
    "Heli %": probabilities_stored[:, 2], 
    "Pred Label": pred_lab_stored,
    "T/F": check,
}

data_frame = pd.DataFrame(table)

# %% Performance Metrics
accuracy = accuracy_score(true_lab_stored, pred_lab_stored)
print(classification_report(true_lab_stored, pred_lab_stored))
print(confusion_matrix(true_lab_stored, pred_lab_stored))
accuracy = (true_lab_stored == pred_lab_stored).mean()
print(f"Overall Accuracy: {accuracy:.4f}")

# %% Robustness Evaluation with Noise
noise_init = Noise(noise_type = 'gaussian')

state_noise = [0.03, 0.05, 0.10, 0.15] 
deriv_noise = [0.05, 0.10, 0.15, 0.20]      

true_class_noi = np.zeros((len(label_test), len(state_noise)))
predictions_noi = np.zeros((len(label_test), len(state_noise)))

for i, (s_noi, d_noi) in enumerate(zip(state_noise, deriv_noise)):
    x_test_noise = noise_init.generate_noise(data=x_test_norm, mean=0, std=s_noi)
    dx_test_noise = noise_init.generate_noise(data=dx_test_norm, mean=0, std=d_noi)

    test_dataset_noi = UAVDataset(x_test_noise, dx_test_noise, label_test, num_classes=classes)
    test_loader_noi = DataLoader(test_dataset_noi, batch_size=1000, shuffle=False)

    for j, (X_batch, Y_batch) in enumerate(test_loader_noi):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        x = X_batch[:, :12]
        labels = X_batch[:, 12:]  

        pred_class, _, confidences = classifier.classify(x=x, dx=Y_batch)

        true_class_noi[j, i] = torch.argmax(labels, dim=1).cpu().numpy()[-1]
        predictions_noi[j, i] = pred_class

# %% Performance Metrics under Noise
accuracy_noi = []
class_report = []

for i in range(len(true_class_noi[1])):
    acc_check = accuracy_score(true_class_noi[:, i], predictions_noi[:, i])
    report_dict = classification_report(true_class_noi[:, i],
                                    predictions_noi[:, i],
                                    output_dict=True)

    accuracy_noi.append(acc_check)
    class_report.append(pd.DataFrame(report_dict).transpose())

accuracy = accuracy_score(true_class_noi, predictions_noi)
print(classification_report(true_lab_stored, pred_lab_stored))
print(confusion_matrix(true_lab_stored, pred_lab_stored))
accuracy = (true_lab_stored == pred_lab_stored).mean()
print(f"Overall Accuracy: {accuracy:.4f}")

# %%

