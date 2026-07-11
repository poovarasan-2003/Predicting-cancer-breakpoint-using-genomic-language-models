import os
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import random

def validate_and_split(pos_fasta, neg_fasta, output_dir, seed=42):
    print("Loading sequences for validation and splitting...")
    
    data = []
    for record in SeqIO.parse(pos_fasta, "fasta"):
        data.append({'id': record.id, 'seq': str(record.seq).upper(), 'label': 1})
    pos_count = len(data)
    
    neg_data = []
    for record in SeqIO.parse(neg_fasta, "fasta"):
        neg_data.append({'id': record.id, 'seq': str(record.seq).upper(), 'label': 0})
        
    print(f"Loaded {pos_count} positive and {len(neg_data)} negative 100kb windows.")
    print("Undersampling negatives to match positives to prevent memory/training bottleneck on GPU...")
    random.seed(seed)
    neg_sampled = random.sample(neg_data, min(pos_count, len(neg_data)))
    
    df = pd.DataFrame(data + neg_sampled)
    df = df.drop_duplicates(subset=['seq'])
    
    print("Splitting dataset (70/10/20 stratified)...")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=(2/3), stratify=temp_df['label'], random_state=seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    def save_to_fasta(df_split, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            for _, row in df_split.iterrows():
                f.write(f">{row['id']}|label={row['label']}\n{row['seq']}\n")
    
    save_to_fasta(train_df, "train.fasta")
    save_to_fasta(val_df, "val.fasta")
    save_to_fasta(test_df, "test.fasta")
    
    print(f"Final Counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    POS_FASTA = os.path.join(BASE_DIR, "Data/Intermediates/positive_sequences.fasta")
    NEG_FASTA = os.path.join(BASE_DIR, "Data/Intermediates/negative_sequences.fasta")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Data/Final_Dataset")
    
    if not os.path.exists(POS_FASTA) or not os.path.exists(NEG_FASTA):
        print("Error: Intermediate fasta files not found. Run extract_positive.py and extract_negative.py first.")
    else:
        validate_and_split(POS_FASTA, NEG_FASTA, OUTPUT_DIR)