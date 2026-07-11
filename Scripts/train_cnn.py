import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from tqdm import tqdm
from util import GenomicDataset, get_base_dir

class DualStrandCNN(nn.Module):
    def __init__(self, seq_length=1024):
        super(DualStrandCNN, self).__init__()

        self.conv1 = nn.Conv1d(4, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        

        flat_size = 256 * 16 
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_one_strand(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):

        feat_fwd = self.forward_one_strand(x)

        x_rev = torch.flip(x, dims=[2])
        x_rev = x_rev[:, [3, 2, 1, 0], :]
        feat_rev = self.forward_one_strand(x_rev)

        feat_combined = torch.max(feat_fwd, feat_rev)
        
        return self.classifier(feat_combined)

def train_model():
    BASE_DIR = get_base_dir()
    DATA_DIR = os.path.join(BASE_DIR, "Data/Final_Dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Models/cnn_dual_strand.pth")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = GenomicDataset(os.path.join(DATA_DIR, "train.fasta"), use_onehot=True, augment=True)
    val_ds = GenomicDataset(os.path.join(DATA_DIR, "val.fasta"), use_onehot=True, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DualStrandCNN(seq_length=1024).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_auc = 0
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        
        print(f"Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with AUC: {val_auc:.4f}")

if __name__ == "__main__":
    train_model()
