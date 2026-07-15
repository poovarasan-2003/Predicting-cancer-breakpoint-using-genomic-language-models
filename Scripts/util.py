import os
import numpy as np
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from Bio.Seq import Seq
import random

def encode_dna_onehot(seq):
    """Standard one-hot encoding."""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = [mapping.get(base.upper(), [0,0,0,0]) for base in seq]
    return np.array(encoded, dtype=np.float32)

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    return str(Seq(seq).reverse_complement())

class GenomicDataset(Dataset):
    def __init__(self, samples, tokenizer=None, max_length=100000, use_onehot=False, augment=False):
        """
        Samples is a list of dicts: {'id': ..., 'seq': ..., 'label': ...}
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_onehot = use_onehot
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['seq']
        label = sample['label']
        
        # Enforce max_length truncation to prevent OOM or CNN flat_size crashes
        # when reading from the raw 100kb FASTA files.
        if self.max_length and len(seq) > self.max_length:
            seq = seq[:self.max_length]
            
        if self.augment and random.random() > 0.5:
            seq = reverse_complement(seq)
            
        if self.use_onehot:
            encoded = encode_dna_onehot(seq)
            encoded = np.transpose(encoded) 
            return torch.tensor(encoded), torch.tensor(label, dtype=torch.float32)
        
        elif self.tokenizer:
            encoded = self.tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return encoded["input_ids"].squeeze(0), torch.tensor(label, dtype=torch.float32)
        
        else:
            return seq, torch.tensor(label, dtype=torch.float32)

def load_fasta_to_samples(fasta_path, label_key='l99'):
    """
    Parses the new header format: >win_chr1:0-100000|density=0.000000|l99=0|l99.5=0|l99.9=0|lind=0
    label_key can be 'l99', 'l99.5', 'l99.9', or 'lind'.
    """
    samples = []
    print(f"Loading samples from {fasta_path} for target {label_key}...")
    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description
        # Extract label from header
        parts = desc.split('|')
        label_map = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=')
                label_map[k] = v
        
        label = int(label_map.get(label_key, 0))
        samples.append({
            'id': record.id,
            'seq': str(record.seq).upper(),
            'label': label,
            'chrom': record.id.split('_')[1].split(':')[0] # win_chr1:0-100000 -> chr1
        })
    print(f"-> Loaded {len(samples)} samples.")
    return samples

def get_base_dir():
    """Helper to find project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
