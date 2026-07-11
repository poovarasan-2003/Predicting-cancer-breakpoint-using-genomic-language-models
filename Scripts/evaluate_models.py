import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from util import GenomicDataset, get_base_dir
from train_cnn import DualStrandCNN

def evaluate_model(name, model, dataloader, device, is_hf=False):
    print(f"Evaluating {name}...")
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if is_hf:
                logits = model(x).logits.squeeze()
            else:
                logits = model(x).squeeze()
            
            if logits.dim() == 0: logits = logits.unsqueeze(0)
            
            if name == "CNN":
                # CNN has Sigmoid in its forward pass
                probs = logits
            else:
                # Foundation models use BCEWithLogitsLoss in training, so they need sigmoid here
                probs = torch.sigmoid(logits)
                
            preds.extend(probs.cpu().numpy())
            labels.extend(y.cpu().numpy())
            
    auc = roc_auc_score(labels, preds)
    binary_preds = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(labels, binary_preds)
    prec = precision_score(labels, binary_preds)
    rec = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    
    return {
        "Model": name,
        "AUC": auc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

def run_evaluation():
    BASE_DIR = get_base_dir()
    DATA_DIR = os.path.join(BASE_DIR, "Data/Final_Dataset")
    TEST_FASTA = os.path.join(DATA_DIR, "test.fasta")
    device = torch.device("cpu") 
    
    results = []
    
    # 1. Evaluate CNN
    print("\n--- Evaluating CNN ---")
    cnn_model = DualStrandCNN(seq_length=1024)
    cnn_path = os.path.join(BASE_DIR, "Models/cnn_dual_strand.pth")
    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.to(device)
        test_ds = GenomicDataset(TEST_FASTA, use_onehot=True, augment=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        results.append(evaluate_model("CNN", cnn_model, test_loader, device))
    else:
        print("CNN model weights not found.")

    # 2. Evaluate HyenaDNA
    print("\n--- Evaluating HyenaDNA ---")
    hyena_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    hyena_path = os.path.join(BASE_DIR, "Models/hyenadna.pth")
    if os.path.exists(hyena_path):
        tokenizer = AutoTokenizer.from_pretrained(hyena_name, trust_remote_code=True, use_fast=False)
        hyena_model = AutoModelForSequenceClassification.from_pretrained(hyena_name, trust_remote_code=True, num_labels=1)
        hyena_model.load_state_dict(torch.load(hyena_path, map_location=device))
        hyena_model.to(device)
        test_ds = GenomicDataset(TEST_FASTA, tokenizer=tokenizer, augment=False)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
        results.append(evaluate_model("HyenaDNA", hyena_model, test_loader, device, is_hf=True))
    else:
        print("HyenaDNA model weights not found.")

    # 3. Evaluate Nucleotide Transformer
    print("\n--- Evaluating Nucleotide Transformer ---")
    nt_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    nt_path = os.path.join(BASE_DIR, "Models/nucleotide_transformer.pth")
    if os.path.exists(nt_path):
        tokenizer = AutoTokenizer.from_pretrained(nt_name, trust_remote_code=True)
        nt_model = AutoModelForSequenceClassification.from_pretrained(nt_name, trust_remote_code=True, num_labels=1)
        nt_model.load_state_dict(torch.load(nt_path, map_location=device))
        nt_model.to(device)
        test_ds = GenomicDataset(TEST_FASTA, tokenizer=tokenizer, augment=False)
        test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
        results.append(evaluate_model("NT-500M", nt_model, test_loader, device, is_hf=True))
    else:
        print("Nucleotide Transformer model weights not found.")

    # Summary
    if results:
        res_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("FINAL TEST RESULTS COMPARISON")
        print("="*50)
        print(res_df.to_string(index=False))
        
        output_csv = os.path.join(BASE_DIR, "Data/Final_Dataset/final_results.csv")
        res_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    run_evaluation()
