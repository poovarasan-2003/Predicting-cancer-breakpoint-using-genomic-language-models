import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import time
from util import GenomicDataset, load_fasta_to_samples, get_base_dir
import random

def get_predictions(model, dataloader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="  Predicting", leave=False):
            x = x.to(device)
            logits = model(x).logits.squeeze()
            if logits.dim() == 0: logits = logits.unsqueeze(0)
            probs.extend(torch.sigmoid(logits).cpu().float().numpy().flatten())
    return np.array(probs)

def train_one_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=16):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    print(f"  Starting epoch with {len(dataloader)} steps...")
    pbar = tqdm(dataloader, desc="  Training Epoch", leave=False)
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device).unsqueeze(1).float()
        
        outputs = model(x).logits
        loss = criterion(outputs, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
        if i < 5 or (i + 1) % 10 == 0:
            pbar.set_postfix({'Loss': f"{loss.item() * accumulation_steps:.4f}"})
            
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)

def run_pu_learning(label_key='l99'):
    start_time = time.time()
    BASE_DIR = get_base_dir()
    FASTA_PATH = os.path.join(BASE_DIR, "Data/Intermediates/all_windows_100kb.fasta")
    
    model_filename = f"nt_pu_paper_{label_key}.pth"
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "Models", model_filename)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training NT PU model for label: {label_key}")

    model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    all_samples = load_fasta_to_samples(FASTA_PATH, label_key=label_key)
    
    chroms = sorted(list(set(s['chrom'] for s in all_samples)))
    random.shuffle(chroms)
    train_chroms = chroms[:int(0.7 * len(chroms))]
    test_chroms = chroms[int(0.7 * len(chroms)):]
    
    train_samples = [s for s in all_samples if s['chrom'] in train_chroms]
    test_samples = [s for s in all_samples if s['chrom'] in test_chroms]
    
    initial_positives = [s for s in train_samples if s['label'] == 1]
    initial_unlabeled = [s for s in train_samples if s['label'] == 0]
    
    print(f"Initial Training Set: P={len(initial_positives)}, U={len(initial_unlabeled)}")
    
    epsilon = 0.03
    max_k = 5
    
    current_P = initial_positives[:]
    current_RN = []
    current_RP = []
    current_U = initial_unlabeled[:]
    
    SEQ_MAX_LEN = 1024 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        num_labels=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    prev_rn_count = 0
    
    for k in range(max_k):
        print(f"\n--- PU Iteration {k+1}/{max_k} ---")
        
        train_data = []
        for s in current_P: train_data.append({'seq': s['seq'], 'label': 1})
        for s in current_RP: train_data.append({'seq': s['seq'], 'label': 1})
        for s in current_RN: train_data.append({'seq': s['seq'], 'label': 0})
        
        num_positives = len(current_P) + len(current_RP)
        u_sample_size = min(len(current_U), max(1000, num_positives * 5))
        sampled_U = random.sample(current_U, u_sample_size)
        
        for s in sampled_U: train_data.append({'seq': s['seq'], 'label': 0})
        
        train_ds = GenomicDataset(train_data, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        
        torch.cuda.empty_cache()
        for sub_epoch in range(3):
            print(f"  Training sub-epoch {sub_epoch+1}/3...")
            train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps=16)
        
        p_ds = GenomicDataset(current_P, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        p_loader = DataLoader(p_ds, batch_size=8, shuffle=False)
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
            
        r_up = max(0.0, min(1.0, r_up))
        r_low = max(0.0, min(1.0, r_low))
            
        print(f"Bounds: r_up={r_up:.4f}, r_low={r_low:.4f}")
        
        unlabeled_ds = GenomicDataset(current_U, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=8, shuffle=False)
        u_probs = get_predictions(model, unlabeled_loader, device)
        
        effective_r_low = min(r_low, 0.4)
        effective_r_up = max(r_up, 0.6)
        
        new_RN_indices = np.where(u_probs < effective_r_low)[0]
        new_RP_indices = np.where(u_probs > effective_r_up)[0]
        
        new_RN = [current_U[i] for i in new_RN_indices]
        new_RP = [current_U[i] for i in new_RP_indices]
        
        print(f"New labeled: RN={len(new_RN)}, RP={len(new_RP)}")
        
        if len(new_RN) > prev_rn_count or len(new_RN) < len(initial_positives):
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
            
        current_RN.extend(new_RN)
        current_RP.extend(new_RP)
        
        indices_to_remove = set(new_RN_indices.tolist() + new_RP_indices.tolist())
        current_U = [s for i, s in enumerate(current_U) if i not in indices_to_remove]
        
        prev_rn_count = len(new_RN)
        
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    test_ds = GenomicDataset(test_samples, tokenizer=tokenizer, max_length=SEQ_MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    test_probs = get_predictions(model, test_loader, device)
    test_labels = np.array([s['label'] for s in test_samples])
    
    roc_auc = roc_auc_score(test_labels, test_probs)
    precision, recall, _ = precision_recall_curve(test_labels, test_probs)
    pr_auc = auc(recall, precision)
    
    print(f"\nFinal Results:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
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
    parser = argparse.ArgumentParser(description="Run PU Learning for NT")
    parser.add_argument("--label", type=str, default="l99", 
                        choices=["l99", "l99.5", "l99.9", "lind"])
    args = parser.parse_args()
    
    run_pu_learning(label_key=args.label)
