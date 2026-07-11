import sys
import os

try:
    import mamba_ssm
    import causal_conv1d
    print("Mamba CUDA kernels available.")
except ImportError:
    print("Mamba CUDA kernels not found! Falling back to reference implementations (VERY SLOW).")
    from unittest.mock import MagicMock
    sys.modules["selective_scan_cuda"] = MagicMock()
    sys.modules["causal_conv1d_cuda"] = MagicMock()
    
    import mamba_ssm.ops.selective_scan_interface as mssi
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    mssi.selective_scan_fn = selective_scan_ref
    
    import causal_conv1d.causal_conv1d_interface as cci
    def causal_conv1d_ref(x, weight, bias=None, activation=None):
        import torch.nn.functional as F
        dim = x.shape[1]
        width = weight.shape[1]
        x_padded = F.pad(x, (width - 1, 0))
        out = F.conv1d(x_padded, weight.unsqueeze(1), bias=bias, groups=dim)
        if activation == "silu":
            out = F.silu(out)
        return out
    cci.causal_conv1d_fn = causal_conv1d_ref

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from util import GenomicDataset, get_base_dir
from torch.cuda.amp import autocast, GradScaler

class CaduceusClassifier(nn.Module):
    def __init__(self, model_name):
        super(CaduceusClassifier, self).__init__()

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        hidden_size = self.backbone.config.d_model
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):
        outputs = self.backbone(input_ids)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1) 
        return self.classifier(pooled_output)

def train_model():
    BASE_DIR = get_base_dir()
    DATA_DIR = os.path.join(BASE_DIR, "Data/Final_Dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Models/caduceus_ps.pth")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading Datasets (100kb sequences)...")
    # Using max_length 100,000 for paper methodology
    train_ds = GenomicDataset(os.path.join(DATA_DIR, "train.fasta"), tokenizer=tokenizer, max_length=100000, augment=True)
    val_ds = GenomicDataset(os.path.join(DATA_DIR, "val.fasta"), tokenizer=tokenizer, max_length=100000, augment=False)
    
    # Very small batch size to fit 100k length on 24GB A5000
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("Initializing Caduceus-PS Model...")
    model = CaduceusClassifier(model_name).to(device)
    
    # Enabling gradient checkpointing to save VRAM if supported by backbone
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient Checkpointing Enabled.")
    
    # BCEWithLogitsLoss is more numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # Accumulate gradients to simulate a larger batch size (e.g. 16)
    accumulation_steps = 16 
    scaler = GradScaler()
    
    best_pr_auc = 0
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                with autocast():
                    logits = model(x).squeeze()
                    
                if logits.dim() == 0: logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits)
                
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        # Calculate PR-AUC as it's the primary metric in the paper for imbalanced datasets
        roc_auc = roc_auc_score(val_labels, val_preds)
        precision, recall, _ = precision_recall_curve(val_labels, val_preds)
        pr_auc = auc(recall, precision)
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
        
        print(f"Loss: {train_loss/len(train_loader):.4f} | Val ROC-AUC: {roc_auc:.4f} | Val PR-AUC: {pr_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    train_model()