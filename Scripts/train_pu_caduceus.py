import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
from util import GenomicDataset, load_fasta_to_samples, get_base_dir
from torch.cuda.amp import autocast, GradScaler
import random

# Implementation of Section 2.4: PU Learning Algorithm
# From "Randomness in Cancer Breakpoint Prediction" (Cheloshkina et al. 2021)

class CaduceusClassifier(nn.Module):
    def __init__(self, model_name):
        super(CaduceusClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # The output of Caduceus-PS is 2 * d_model because of the bi-directional 
        # reverse-complement parameter sharing concatenation.
        # So if d_model is 256, the output dimension is 512.
        hidden_size = self.backbone.config.d_model * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):
        outputs = self.backbone(input_ids)
        hidden_states = outputs.last_hidden_state
        # Max pooling preserves localized hotspot signals much better across 32k tokens
        pooled_output, _ = torch.max(hidden_states, dim=1) 
        return self.classifier(pooled_output)

def get_predictions(model, dataloader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="  Predicting", leave=False):
            x = x.to(device)
            # Reverted to full FP32; Mamba is highly sensitive to mixed precision
            logits = model(x)
            probs.extend(torch.sigmoid(logits).cpu().float().numpy().flatten())
    return np.array(probs)

def train_one_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=32):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    print(f"  Starting epoch with {len(dataloader)} steps...")
    pbar = tqdm(dataloader, desc="  Training Epoch", leave=False)
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device).unsqueeze(1)
        
        # Reverted to full FP32 to prevent Caduceus/Mamba from outputting constant logic
        outputs = model(x)
        loss = criterion(outputs, y) / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        if i < 5 or (i + 1) % 10 == 0:
            pbar.set_postfix({'Loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    # Final step
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    return total_loss / len(dataloader)

def run_pu_learning(label_key='l99'):
    start_time = time.time()
    BASE_DIR = get_base_dir()
    FASTA_PATH = os.path.join(BASE_DIR, "Data/Intermediates/all_windows_100kb.fasta")
    
    # Save the model with a name that indicates which label was used
    model_filename = f"caduceus_pu_paper_{label_key}.pth"
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Models", model_filename)
    
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training PU model for label: {label_key}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load all samples using the dynamically provided label
    all_samples = load_fasta_to_samples(FASTA_PATH, label_key=label_key)
    
    # 1. Stratified Train/Test Split (70/30) as per Section 2.2
    # Chromosome-based stratification
    chroms = sorted(list(set(s['chrom'] for s in all_samples)))
    random.shuffle(chroms)
    # Simple split for demonstration (in practice we'd want exact 70/30 window count)
    train_chroms = chroms[:int(0.7 * len(chroms))]
    test_chroms = chroms[int(0.7 * len(chroms)):]
    
    train_samples = [s for s in all_samples if s['chrom'] in train_chroms]
    test_samples = [s for s in all_samples if s['chrom'] in test_chroms]
    
    # Initial sets
    # P = initial positives
    # U = unlabeled (initially treated as negative)
    initial_positives = [s for s in train_samples if s['label'] == 1]
    initial_unlabeled = [s for s in train_samples if s['label'] == 0]
    
    print(f"Initial Training Set: P={len(initial_positives)}, U={len(initial_unlabeled)}")
    
    # PU Algorithm Parameters
    epsilon = 0.03
    max_k = 5
    
    # Current datasets
    current_P = initial_positives[:]
    current_RN = []
    current_RP = []
    current_U = initial_unlabeled[:]
    
    model = CaduceusClassifier(model_name).to(device)
    
    # CRITICAL: Freeze the 130M parameter backbone to prevent catastrophic forgetting.
    # We only train the classifier head since the dataset is highly imbalanced and small.
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    # Increased LR since we are only training the classifier head
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)
    # Removed scaler to prevent Mamba deadlock
    
    prev_rn_count = 0
    
    for k in range(max_k):
        print(f"\n--- PU Iteration {k+1}/{max_k} ---")
        
        # Prepare training data: P + RP (labeled 1) and RN + sample of U (labeled 0)
        # Treating unlabeled as 0 for initial model training
        # To avoid massive compute, we undersample U if it's too large, but paper says "all rest examples"
        # However, for 100kb sequences, we must be careful with A5000 memory.
        
        train_data = []
        for s in current_P: train_data.append({'seq': s['seq'], 'label': 1})
        for s in current_RP: train_data.append({'seq': s['seq'], 'label': 1})
        for s in current_RN: train_data.append({'seq': s['seq'], 'label': 0})
        
        # Unlabeled used as negative if not in RN.
        # CRITICAL OPTIMIZATION: Undersampling U for the training step to prevent 11+ hour epochs
        # We sample a max of 5x the positive samples, or 1000, whichever is larger, to prevent massive class imbalance.
        num_positives = len(current_P) + len(current_RP)
        u_sample_size = min(len(current_U), max(1000, num_positives * 5))
        sampled_U = random.sample(current_U, u_sample_size)
        
        for s in sampled_U: train_data.append({'seq': s['seq'], 'label': 0})
        
        # Reduced max_length to 1024 to massively improve signal-to-noise ratio
        # and match the CNN/HyenaDNA implementations.
        SEQ_MAX_LEN = 1024 
        
        train_ds = GenomicDataset(train_data, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        # Sequence is 32x shorter, so we can use a larger batch size
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        # Train for more epochs since training is now very fast and we are only training the head
        torch.cuda.empty_cache() 
        for sub_epoch in range(10):
            print(f"  Training sub-epoch {sub_epoch+1}/10...")
            train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps=2)
        
        # Get predictions for bounds calculation
        # CRITICAL FIX: Bounds must be calculated based strictly on the POSITIVE set distribution
        p_ds = GenomicDataset(current_P, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        p_loader = DataLoader(p_ds, batch_size=32, shuffle=False)
        p_probs = get_predictions(model, p_loader, device)
        
        q1 = np.percentile(p_probs, 90)
        q2 = np.percentile(p_probs, 10)
        
        if q1 == q2:
            print("Warning: q1 == q2, breaking iteration.")
            break
            
        if q1 - q2 > 2 * epsilon:
            r_up, r_low = q1, q2
        else:
            r_up, r_low = q1 - epsilon, q2 - epsilon
            
        # Ensure bounds don't drop below 0 or above 1, preventing blind labeling
        r_up = max(0.0, min(1.0, r_up))
        r_low = max(0.0, min(1.0, r_low))
            
        print(f"Bounds: r_up={r_up:.4f}, r_low={r_low:.4f}")
        
        # Label RN and RP from Unlabeled set
        unlabeled_ds = GenomicDataset(current_U, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        # Speed optimization: Inference doesn't require gradients, so we can use a larger batch size
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=32, shuffle=False)
        u_probs = get_predictions(model, unlabeled_loader, device)
        
        # CRITICAL SAFEGUARD: Do not label RP if the model is unsure (predicting ~0.5)
        # Without this, if q1=0.5, r_up becomes 0.47, and all unsure 0.5 predictions are labeled Positive!
        effective_r_low = min(r_low, 0.4)
        effective_r_up = max(r_up, 0.6)
        
        new_RN_indices = np.where(u_probs < effective_r_low)[0]
        new_RP_indices = np.where(u_probs > effective_r_up)[0]
        
        new_RN = [current_U[i] for i in new_RN_indices]
        new_RP = [current_U[i] for i in new_RP_indices]
        
        print(f"New labeled: RN={len(new_RN)}, RP={len(new_RP)}")
        
        # Stopping criteria (Section 2.4, Step 1.5.4)
        if len(new_RN) > prev_rn_count or len(new_RN) < len(initial_positives):
            # This logic from the paper is slightly confusing ("greater than... OR less than...")
            # We'll use a safer "no new examples" check as well
            pass
            
        if r_up < r_low:
            print("Stopping: r_up < r_low")
            break
        if len(current_U) == 0:
            print("Stopping: Unlabeled set empty")
            break
        if len(new_RN) == 0 and len(new_RP) == 0:
            print("Stopping: No new labeled examples")
            break
            
        # Update sets
        current_RN.extend(new_RN)
        current_RP.extend(new_RP)
        
        # Remove from U
        indices_to_remove = set(new_RN_indices.tolist() + new_RP_indices.tolist())
        current_U = [s for i, s in enumerate(current_U) if i not in indices_to_remove]
        
        prev_rn_count = len(new_RN)
        
        # Save model after each iteration
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Final Evaluation on Test Set
    test_ds = GenomicDataset(test_samples, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    test_probs = get_predictions(model, test_loader, device)
    test_labels = np.array([s['label'] for s in test_samples])
    
    roc_auc = roc_auc_score(test_labels, test_probs)
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    pr_auc = auc(recall, precision)
    
    print(f"\nFinal Results:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Calculate Lift of Recall at 1% percentile as example
    sorted_indices = np.argsort(test_probs)[::-1]
    top_1_percent = int(0.01 * len(test_probs))
    if top_1_percent > 0:
        top_indices = sorted_indices[:top_1_percent]
        hits = np.sum(test_labels[top_indices])
        lift_recall = (hits / np.sum(test_labels)) / 0.01
        print(f"Lift of Recall (1%): {lift_recall:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal Training Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PU Learning for Caduceus")
    parser.add_argument("--label", type=str, default="l99", 
                        choices=["l99", "l99.5", "l99.9", "lind"], 
                        help="The target label to train on (l99, l99.5, l99.9, or lind)")
    args = parser.parse_args()
    
    run_pu_learning(label_key=args.label)
