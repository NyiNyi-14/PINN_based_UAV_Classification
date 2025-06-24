# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR

from PINN import PINN

# %%
# ================== ENHANCED TRAINER ==================
class PINN_Classifier:
    def __init__(self, input_dim, output_dim, lr, delta_loss, interval):
        self.model = PINN(input_dim, output_dim)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience = 20, min_lr = 1e-6,
        )
        self.delta_loss = delta_loss
        self.interval = interval
        self.best_model_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def compute_physics_loss(self, y_pred):
        if y_pred.shape[0] > 1:
            phy_loss = y_pred[1:, :] - y_pred[:-1, :]
            return torch.mean(phy_loss**2)
        return torch.tensor(0.0, device=y_pred.device)

    def hybrid_loss(self, y_pred, y_true):
        mse_loss = self.loss_func(y_pred, y_true)
        physics_loss = self.compute_physics_loss(y_pred)
        return mse_loss + 0.2 * physics_loss

    def train(self, train, validate, num_epochs):
        train_losses, val_losses = [], []
        print_interval = 10
        start_time = time.time()
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

        phase1_end = int(0.3 * num_epochs)
        phase2_end = int(0.7 * num_epochs)

        phase1 = ConstantLR(self.optimizer, 
                            factor = 1.0,
                            total_iters = phase1_end)
        
        phase2 = LinearLR(self.optimizer, 
                          start_factor = 1.0, 
                          end_factor = 0.1, 
                          total_iters = phase2_end - phase1_end)
        
        phase3 = LinearLR(self.optimizer, 
                            start_factor = 0.1, 
                            end_factor = 0.01, 
                            total_iters = num_epochs - phase2_end)
        
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[phase1, phase2, phase3], 
                                      milestones=[phase1_end, phase2_end])
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            for X_batch, Y_batch in train:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                self.optimizer.zero_grad()
                Y_pred = self.model(X_batch)
                loss = self.hybrid_loss(Y_pred, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train) 
            train_losses.append(epoch_train_loss) 

            # Validation phase
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in validate:
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                    Y_pred = self.model(X_batch)
                    loss = self.hybrid_loss(Y_pred, Y_batch)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(validate)
            val_losses.append(epoch_val_loss)

            # Save best model state
            if self.best_model_state is None or epoch_val_loss < min(val_losses[:-1]):
                self.best_model_state = self.model.state_dict()

            # Learning rate schedule
            # self.scheduler.step(epoch_val_loss)
            self.scheduler.step()  

            if (epoch + 1) % print_interval == 0:
                total_time = time.time() - start_time
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                    f"Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
                    f"Time: {total_time // 60:.0f} min {total_time % 60:.2f} sec")
           
            # Early stopping check
            if epoch >= self.interval and epoch % self.interval == 0:
                prev_avg = np.mean(val_losses[epoch - self.interval:epoch])
                curr_avg = np.mean(val_losses[epoch - self.interval + 1:epoch + 1])
                if curr_avg > (prev_avg - self.delta_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Save best model
        model_path = os.path.join(save_dir, "Pinn_model.pth")
        torch.save(self.best_model_state, model_path)

        # Save losses
        np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(save_dir, "val_losses.npy"), np.array(val_losses))

        print(f"Model and loss history saved to: {save_dir}")

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Losses")
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(save_dir, "loss_plot.png"))
        plt.close()

    # ================== IMPROVED CLASSIFICATION ==================
    def classify(self, x, u, dx, num_classes=3):
        self.model.eval()

        UAV_LABELS = {
        0: "Quadcopter",
        1: "Fixed-wing",
        2: "Helicopter"
        }

        # Convert to tensors (vectorized)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device = self.device)
        u_tensor = torch.as_tensor(u, dtype=torch.float32, device = self.device)
        dx_tensor = torch.as_tensor(dx, dtype=torch.float32, device = self.device)

        # Generate all class labels at once
        labels = torch.eye(num_classes, device=x_tensor.device)

        # Create batch for all classes
        inputs = torch.cat([
            x_tensor.repeat_interleave(num_classes, 0),
            u_tensor.repeat_interleave(num_classes, 0),
            labels.repeat(x.shape[0], 1)
        ], dim=1)

        with torch.no_grad():
            dx_pred = self.model(inputs).view(-1, num_classes, dx.shape[1])
            losses = torch.mean((dx_pred - dx_tensor.unsqueeze(1))**2, dim=2)
            mean_loss = torch.mean(losses, dim=0)  # [C]
            probs = torch.softmax(-mean_loss * 10, dim=0) * 100
        
        loss_vals = mean_loss.cpu().numpy()
        conf_vals = probs.cpu().numpy().round(2)

        print(f"Quad Loss: {loss_vals[0]:.4f}")
        print(f"FW   Loss: {loss_vals[1]:.4f}")
        print(f"Heli Loss: {loss_vals[2]:.4f}")

        print(f"Confidence (%): Quad={conf_vals[0]:.2f}%, FW={conf_vals[1]:.2f}%, Heli={conf_vals[2]:.2f}%")
        print(f"Predicted UAV Type: {UAV_LABELS[torch.argmin(mean_loss).item()]}")

        return (
            torch.argmin(mean_loss).item(),              # predicted class (int)
            loss_vals,                     # avg MSE loss per class
            conf_vals                 # softmax confidence per class
        )

    # ================== NEW PHYSICS VALIDATION ==================
    def validate_physics(self, test_loader):
        """Check physical consistency of predictions"""
        self.model.eval()
        violations = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                Y_pred = self.model(X_batch)
                # Example check: Energy should be below threshold
                energy = torch.sum(Y_pred[:, :3]**2, dim=1)  # Kinetic energy
                violations += torch.sum(energy > 1.0).item()
        print(f"Physics violations: {violations}/{len(test_loader.dataset)}")

# %%