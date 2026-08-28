import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class OPFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OPFSurrogate(nn.Module):
    def __init__(self, input_dim=22, output_dim=4, v_min=0.94, v_max=1.10):
        super(OPFSurrogate, self).__init__()
        self.v_min = v_min
        self.v_max = v_max
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        z = self.net(x)
        # Scaled sigmoid guarantees output stays strictly within [v_min, v_max]
        return self.v_min + torch.sigmoid(z) * (self.v_max - self.v_min)

def train_model(data_path="dataset.csv", epochs=200, batch_size=32, lr=0.001, v_min=0.94, v_max=1.10):
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # 22 input features (11 P_L, 11 Q_L)
    feature_cols = [c for c in df.columns if c.startswith('P_L') or c.startswith('Q_L')]
    target_cols = ['V_G2', 'V_G3', 'V_G6', 'V_G8']
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Feature Standardization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1e-6
    X_scaled = (X - X_mean) / X_std
    
    np.save("X_mean.npy", X_mean)
    np.save("X_std.npy", X_std)
    
    # Split into 70% Train, 15% Validation, 15% Test
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=15/85, random_state=42)
    
    # Save test set partition for standalone evaluation
    np.save("test_indices.npy", test_idx)
    np.save("X_test.npy", X_scaled[test_idx])
    np.save("X_test_raw.npy", X[test_idx])
    np.save("y_test.npy", y[test_idx])
    
    train_dataset = OPFDataset(X_scaled[train_idx], y[train_idx])
    val_dataset = OPFDataset(X_scaled[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Data split: {len(train_idx)} Train | {len(val_idx)} Validation | {len(test_idx)} Test")
    
    model = OPFSurrogate(input_dim=len(feature_cols), output_dim=len(target_cols), v_min=v_min, v_max=v_max)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting Neural Network Training...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] - Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f} | Best Val MSE: {best_val_loss:.6f}")
            
    print(f"\nTraining Complete! Best Validation MSE Loss: {best_val_loss:.6f}")
    print("Best model weights saved to 'best_model.pth'.")
    
    # Plot and save Training & Validation Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', color='#2563eb', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#dc2626', linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('AI Surrogate Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300)
    plt.close()
    print("Loss curve saved to 'training_loss.png'.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_model()
