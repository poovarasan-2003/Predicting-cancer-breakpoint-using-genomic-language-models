import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm
from util import GenomicDataset, get_base_dir

def train_model():
    BASE_DIR = get_base_dir()
    DATA_DIR = os.path.join(BASE_DIR, "Data/Final_Dataset")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Models/hyenadna.pth")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        num_labels=1
    ).to(device)

    # Load Data
    train_ds = GenomicDataset(os.path.join(DATA_DIR, "train.fasta"), tokenizer=tokenizer, augment=True)
    val_ds = GenomicDataset(os.path.join(DATA_DIR, "val.fasta"), tokenizer=tokenizer, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_val_auc = 0
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            
            outputs = model(x).logits
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).logits.squeeze()
                if logits.dim() == 0: logits = logits.unsqueeze(0)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
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
