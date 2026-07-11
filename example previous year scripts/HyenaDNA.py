import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import create_df, create_df_full, remove_duplicates, create_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from shap.maskers import Masker
from types import SimpleNamespace
import shap
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import psutil
import math
from typing import Optional, Tuple, Union
from einops import rearrange, repeat
from functools import partial
import json
import traceback
from datetime import datetime
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
def parse_arguments():
    parser = argparse.ArgumentParser(description='HyenaDNA Training with SHAP Analysis for Fragile Site Detection')

    # Data parameters
    parser.add_argument('--pos_fasta', type=str,
                       default="/mnt/dc/ta/GLM_HFS/data/merged_fragiles.fa",
                       help='Path to positive sequences FASTA file')
    parser.add_argument('--neg_fasta', type=str,
                       default="/mnt/dc/ta/GLM_HFS/code/reference/negative_training_sequences2.fa",
                       help='Path to negative sequences FASTA file')

    # Data splitting parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--balance_strategy', type=str, default='undersample',
                       choices=['stratified', 'undersample', 'oversample', 'combined'],
                       help='Class balancing strategy (default: undersample)')

    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of HyenaDNA blocks (default: 4)')
    parser.add_argument('--d_inner', type=int, default=512,
                       help='Inner dimension of MLP (default: 512)')
    parser.add_argument('--order', type=int, default=2,
                       help='Order of Hyena operator (default: 2)')
    parser.add_argument('--seq_len', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--vocab_size', type=int, default=8,
                       help='Vocabulary size (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--bidirectional', action='store_true',
                   help='Use bidirectional HyenaDNA architecture')
    parser.add_argument('--bidirectional_strategy', type=str, 
                   default='concat', choices=['concat', 'add', 'max'],
                   help='Strategy for combining bidirectional features')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='Weight decay (default: 1e-3)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0)')

    # SHAP parameters
    parser.add_argument('--enable_shap', action='store_true',
                       help='Enable SHAP analysis (default: False)')
    parser.add_argument('--shap_explainer', type=str, default='partition',
                       choices=['permutation', 'partition', 'kernel'],
                       help='SHAP explainer type (default: partition)')
    parser.add_argument('--shap_background_size', type=int, default=100,
                       help='Number of background sequences for SHAP (default: 100)')
    parser.add_argument('--shap_analysis_size', type=int, default=16,
                       help='Number of sequences to analyze with SHAP (default: 16)')
    parser.add_argument('--shap_tp_samples', type=int, default=5,
                       help='Number of true positive samples for SHAP (default: 5)')
    parser.add_argument('--shap_tn_samples', type=int, default=5,
                       help='Number of true negative samples for SHAP (default: 5)')
    parser.add_argument('--shap_fp_samples', type=int, default=3,
                       help='Number of false positive samples for SHAP (default: 3)')
    parser.add_argument('--shap_fn_samples', type=int, default=3,
                       help='Number of false negative samples for SHAP (default: 3)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for saving results (default: current directory)')
    parser.add_argument('--model_save_name', type=str, default='hyenadna_fragile_sites_model.pth',
                       help='Model save filename (default: hyenadna_fragile_sites_model.pth)')
    parser.add_argument('--history_save_name', type=str, default='training_history.pth',
                       help='Training history save filename (default: training_history.pth)')
    parser.add_argument('--shap_plot_name', type=str, default='hyenadna_shap_analysis.png',
                       help='SHAP visualization save filename (default: hyenadna_shap_analysis.png)')
    parser.add_argument('--shap_report_name', type=str, default='hyenadna_shap_report.txt',
                       help='SHAP report save filename (default: hyenadna_shap_report.txt)')
    parser.add_argument('--shap_data_name', type=str, default='hyenadna_shap_data.pth',
                       help='SHAP data save filename (default: hyenadna_shap_data.pth)')
    parser.add_argument('--enable_visualization', action='store_true',
                        help='Enable UMAP/t-SNE/PCA visualization of embeddings')
    parser.add_argument('--viz_max_samples', type=int, default=2000,
                        help='Maximum samples for visualization (default: 2000)')

    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use for training (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output (default: False)')

    return parser.parse_args()

class BidirectionalHyenaWrapper(nn.Module):
    """
    Wrapper to make the custom HyenaDNAModel bidirectional.
    It takes an initialized HyenaDNAModel instance.
    """
    def __init__(self, hyena_model, num_classes=2, bidirectional_strategy='concat'):
        super(BidirectionalHyenaWrapper, self).__init__()
        
        self.bidirectional_strategy = bidirectional_strategy
    
        self.hyena_model = hyena_model
        
        hidden_size = self.hyena_model.d_model
        
        if bidirectional_strategy == 'concat':
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
            
    def forward(self, input_ids, attention_mask=None):
        fwd_features = self.hyena_model.forward_features(input_ids, attention_mask)
        
        input_ids_rev = torch.flip(input_ids, dims=[1])
        attention_mask_rev = torch.flip(attention_mask, dims=[1]) if attention_mask is not None else None
        rev_features = self.hyena_model.forward_features(input_ids_rev, attention_mask_rev)
        
        if self.bidirectional_strategy == 'concat':
            combined = torch.cat([fwd_features, rev_features], dim=1)
        elif self.bidirectional_strategy == 'add':
            combined = fwd_features + rev_features
        elif self.bidirectional_strategy == 'max':
            combined = torch.max(fwd_features, rev_features)
        else:
            raise ValueError(f"Unknown strategy: {self.bidirectional_strategy}")
            
        # --- Final Classification ---
        logits = self.classifier(combined)
        return logits # Return logits directly for consistency in training loop

class DNAMasker(Masker):
    """Custom masker for DNA sequences"""
    def __init__(self, mask_value="N"):
        self.mask_value = mask_value
        self._shape = None
        self._invariants = None
        
    def __call__(self, x, mask=None):
        """Mask DNA sequences"""
        if isinstance(x, str):
            x = [x]
        
        if mask is None:
            return x
        
        masked_sequences = []
        for seq in x:
            if isinstance(seq, str):
                seq_list = list(seq)
                for i, m in enumerate(mask):
                    if i < len(seq_list) and not m:
                        seq_list[i] = self.mask_value
                masked_sequences.append(''.join(seq_list))
            else:
                masked_sequences.append(seq)
        
        return masked_sequences

def extract_embeddings_from_dataloader(model, dataloader, device, max_samples=None):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_predictions = []

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # Access the forward_features method from the wrapped model
            # and pass the attention mask
            if hasattr(model, 'hyena_model'):
                # For BidirectionalHyenaWrapper
                embeddings = model.hyena_model.forward_features(input_ids, attention_mask)
            else:
                # For a regular HyenaDNAModel
                embeddings = model.forward_features(input_ids, attention_mask)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
            _, predicted = torch.max(logits, 1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

            sample_count += len(labels)
            if max_samples and sample_count >= max_samples:
                break

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    if max_samples:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]
        predictions = predictions[:max_samples]

    return embeddings, labels, predictions

def apply_dimensionality_reduction(embeddings, method='umap', n_components=2, 
                                  random_state=42, **kwargs):
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if method.lower() == 'umap':
        # UMAP parameters for genomic data
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            metric=kwargs.get('metric', 'euclidean'),
            n_epochs=kwargs.get('n_epochs', 200)
        )
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        
    elif method.lower() == 'tsne':
        # t-SNE parameters for genomic data
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=kwargs.get('perplexity', 30),
            learning_rate=kwargs.get('learning_rate', 200),
            n_iter=kwargs.get('n_iter', 1000),
            method=kwargs.get('method', 'barnes_hut'),
            init=kwargs.get('init', 'pca')
        )
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        
    elif method.lower() == 'pca':
        # PCA parameters
        reducer = PCA(
            n_components=n_components,
            random_state=random_state,
            svd_solver=kwargs.get('svd_solver', 'auto')
        )
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        
        # Print explained variance for PCA
        print(f"PCA Explained Variance Ratio: {reducer.explained_variance_ratio_}")
        print(f"Total Variance Explained: {sum(reducer.explained_variance_ratio_):.4f}")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'umap', 'tsne', or 'pca'")
    
    return reduced_embeddings


class HyenaDNATokenizer:
    """
    Character-level tokenizer for HyenaDNA
    """
    def __init__(self):
        self.char_to_id = {
            'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
            'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4,
            '<pad>': 5, '<unk>': 6
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.pad_token_id = 5
        self.unk_token_id = 6
    
    def encode(self, sequence, max_length=1024, padding='max_length', truncation=True, **kwargs):
        """Encode DNA sequence to token IDs"""
        sequence = sequence.upper()
        token_ids = [self.char_to_id.get(char, self.unk_token_id) for char in sequence]
        
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        attention_mask = [1] * len(token_ids)
        
        if padding == 'max_length' and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            token_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def __call__(self, sequences, return_tensors=None, **kwargs):
        """Make tokenizer callable with return_tensors support"""
        if isinstance(sequences, str):
            result = self.encode(sequences, **kwargs)
        else:
            result = {
                'input_ids': [],
                'attention_mask': []
            }
            for seq in sequences:
                encoded = self.encode(seq, **kwargs)
                result['input_ids'].append(encoded['input_ids'])
                result['attention_mask'].append(encoded['attention_mask'])
        
        if return_tensors == "pt":
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result

def fftconv(u, k, D):
    """FFT-based convolution for the Hyena operator"""
    L = u.shape[1]
    k_f = torch.fft.rfft(k, n=2*L, dim=0)
    u_f = torch.fft.rfft(u.float(), n=2*L, dim=1)
    y_f = u_f * k_f
    y = torch.fft.irfft(y_f, n=2*L, dim=1)[:, :L, :]
    return y.to(u.dtype)

class HyenaFilter(torch.nn.Module):
    """Implicit long convolution filter for Hyena operator"""
    def __init__(self, d_model, order=2, seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.seq_len = seq_len

        self.w0 = torch.nn.Parameter(torch.randn(d_model))
        self.w1 = torch.nn.Parameter(torch.randn(d_model))
        self.w2 = torch.nn.Parameter(torch.randn(d_model))
        self.pos_emb = torch.nn.Parameter(torch.randn(seq_len, 1))

        self.filter_mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(1, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, d_model)
            ) for _ in range(order)
        ])

    def forward(self, L):
        """Generate implicit filter of length L"""
        t = torch.linspace(0, 1, L, device=self.w0.device).unsqueeze(1)

        filters = []
        for i in range(self.order):
            h = self.filter_mlps[i](t)
            decay = torch.exp(-self.w0.abs() * t)
            oscillation = torch.cos(2 * math.pi * self.w1.abs() * t + self.w2)
            h = h * decay * oscillation
            filters.append(h)

        return filters

class HyenaOperator(torch.nn.Module):
    """The actual Hyena operator using long convolutions"""
    def __init__(self, d_model, order=2, seq_len=1024, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.order = order

        self.in_proj = torch.nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.hyena_filter = HyenaFilter(d_model, order, seq_len)
        self.short_conv = torch.nn.Conv1d(
            (order + 1) * d_model,
            (order + 1) * d_model,
            kernel_size=3,
            padding=1,
            groups=(order + 1) * d_model
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Forward pass of Hyena operator"""
        B, L, D = x.shape

        xz = self.in_proj(x)
        xz = rearrange(xz, 'b l d -> b d l')
        xz = self.short_conv(xz)
        xz = rearrange(xz, 'b d l -> b l d')

        z, *xs = xz.chunk(self.order + 1, dim=-1)
        z = torch.nn.functional.gelu(z)

        filters = self.hyena_filter(L)

        y = z
        for i in range(self.order):
            y = fftconv(y * xs[i], filters[i], D)

        if mask is not None:
            y = y * mask.unsqueeze(-1).float()

        y = self.out_proj(y)
        y = self.dropout(y)

        return self.norm(x + y)

class HyenaDNABlock(torch.nn.Module):
    """A single HyenaDNA block with Hyena operator and MLP"""
    def __init__(self, d_model, d_inner=None, order=2, seq_len=1024, dropout=0.0):
        super().__init__()
        d_inner = d_inner or 4 * d_model

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, order, seq_len, dropout)

        self.norm2 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_inner),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_model),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = self.hyena(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

class HyenaDNAModel(torch.nn.Module):
    """Complete HyenaDNA model for sequence classification"""
    def forward_features(self, input_ids, attention_mask=None):
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.device != next(self.parameters()).device:
            input_ids = input_ids.to(next(self.parameters()).device)
        
        if attention_mask is not None:
            if isinstance(attention_mask, np.ndarray):
                attention_mask = torch.from_numpy(attention_mask).long()
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            if attention_mask.device != next(self.parameters()).device:
                attention_mask = attention_mask.to(next(self.parameters()).device)
      
        B, L = input_ids.shape
        
        x = self.embed(input_ids)  
        x = x + self.pos_embed[:, :L, :]

    def __init__(
        self,
        vocab_size=8,
        d_model=256,
        n_layers=8,
        d_inner=None,
        order=2,
        seq_len=1024,
        num_classes=2,
        dropout=0.1,
        prenorm=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.prenorm = prenorm
        
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.embed_dropout = torch.nn.Dropout(dropout)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        self.blocks = torch.nn.ModuleList([
            HyenaDNABlock(d_model, d_inner, order, seq_len, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = torch.nn.LayerNorm(d_model) if prenorm else torch.nn.Identity()
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
    def forward_features(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, :L, :]
        x = self.embed_dropout(x)
        
        for block in self.blocks:
            x = block(x, mask=attention_mask)
        
        x = self.norm(x)
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()
            if input_ids.device != next(self.parameters()).device:
                input_ids = input_ids.to(next(self.parameters()).device)
        
        if attention_mask is not None and isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).long()
            if attention_mask.device != next(self.parameters()).device:
                attention_mask = attention_mask.to(next(self.parameters()).device)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-7)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)
            
        return pooled

    def forward(self, input_ids, attention_mask=None):
        pooled = self.forward_features(input_ids, attention_mask)
        logits = self.classifier(pooled)
        return {"logits": logits}

def create_embedding_visualizations(embeddings, labels, predictions=None, 
                                   save_dir="./embedding_visualizations",
                                   sample_names=None):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define color schemes
    colors_true = {0: '#3498db', 1: '#e74c3c'}  # Blue for negative, Red for positive
    colors_pred = {0: '#2ecc71', 1: '#f39c12'}  # Green for negative, Orange for positive
    
    # Apply all three dimensionality reduction methods
    methods = ['UMAP', 'TSNE', 'PCA']
    reduced_embeddings = {}
    
    print("\nApplying dimensionality reduction methods...")
    for method in methods:
        print(f"Computing {method}...")
        reduced = apply_dimensionality_reduction(embeddings, method=method.lower())
        reduced_embeddings[method] = reduced
    
    # Create figure with subplots for all visualizations
    fig = plt.figure(figsize=(24, 16))
    
    # Create 3x2 grid (3 methods x 2 label types)
    for i, method in enumerate(methods):
        reduced = reduced_embeddings[method]
        
        # Plot with true labels
        ax1 = plt.subplot(2, 3, i + 1)
        for label in np.unique(labels):
            mask = labels == label
            ax1.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[colors_true[label]], 
                       label=f'{"Fragile" if label == 1 else "Non-Fragile"}',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        ax1.set_title(f'{method} - True Labels', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{method}1', fontsize=12)
        ax1.set_ylabel(f'{method}2', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot with predicted labels (if available)
        if predictions is not None:
            ax2 = plt.subplot(2, 3, i + 4)
            for label in np.unique(predictions):
                mask = predictions == label
                ax2.scatter(reduced[mask, 0], reduced[mask, 1], 
                           c=[colors_pred[label]], 
                           label=f'Pred: {"Fragile" if label == 1 else "Non-Fragile"}',
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Mark misclassifications
            misclassified = labels != predictions
            if np.any(misclassified):
                ax2.scatter(reduced[misclassified, 0], reduced[misclassified, 1],
                           marker='x', c='red', s=50, label='Misclassified', alpha=0.8)
            
            ax2.set_title(f'{method} - Predicted Labels', fontsize=14, fontweight='bold')
            ax2.set_xlabel(f'{method}1', fontsize=12)
            ax2.set_ylabel(f'{method}2', fontsize=12)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Embeddings Visualization: DNA Fragile Sites Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(save_dir, 'all_methods_visualization.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {combined_path}")
    plt.show()
    
    # Create individual high-quality plots for each method
    for method in methods:
        reduced = reduced_embeddings[method]
        
        # Individual plot with true labels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # True labels
        for label in np.unique(labels):
            mask = labels == label
            ax1.scatter(reduced[mask, 0], reduced[mask, 1], 
                       c=[colors_true[label]], 
                       label=f'{"Fragile" if label == 1 else "Non-Fragile"}',
                       alpha=0.7, s=50, edgecolors='white', linewidth=1.5)
        
        ax1.set_title(f'{method} Embedding Space - True Labels', fontsize=16, fontweight='bold')
        ax1.set_xlabel(f'{method} Component 1', fontsize=14)
        ax1.set_ylabel(f'{method} Component 2', fontsize=14)
        ax1.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Predicted labels
        if predictions is not None:
            for label in np.unique(predictions):
                mask = predictions == label
                ax2.scatter(reduced[mask, 0], reduced[mask, 1], 
                           c=[colors_pred[label]], 
                           label=f'Pred: {"Fragile" if label == 1 else "Non-Fragile"}',
                           alpha=0.7, s=50, edgecolors='white', linewidth=1.5)
            
            # Highlight misclassifications
            misclassified = labels != predictions
            if np.any(misclassified):
                ax2.scatter(reduced[misclassified, 0], reduced[misclassified, 1],
                           marker='x', c='darkred', s=100, label='Misclassified', 
                           alpha=0.9, linewidth=2.5)
            
            ax2.set_title(f'{method} Embedding Space - Predicted Labels', fontsize=16, fontweight='bold')
            ax2.set_xlabel(f'{method} Component 1', fontsize=14)
            ax2.set_ylabel(f'{method} Component 2', fontsize=14)
            ax2.legend(loc='best', fontsize=12, frameon=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        
        plt.suptitle(f'{method} Visualization - DNA Fragile Sites Model', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(save_dir, f'{method.lower()}_visualization.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} visualization to {individual_path}")
        plt.show()
    
    return reduced_embeddings


def create_combined_dataset_plot(all_results, save_dir):
    """
    Create a combined plot showing all datasets side by side.
    """
    methods = ['UMAP', 'TSNE', 'PCA']
    datasets = list(all_results.keys())
    
    for method in methods:
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
        
        if len(datasets) == 1:
            axes = [axes]
        
        for i, dataset_name in enumerate(datasets):
            ax = axes[i]
            reduced = all_results[dataset_name]['reduced_embeddings'][method]
            labels = all_results[dataset_name]['labels']
            
            # Plot with different colors for each class
            colors = ['#3498db', '#e74c3c']  # Blue for negative, Red for positive
            for label in np.unique(labels):
                mask = labels == label
                ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                          c=[colors[label]], 
                          label=f'{"Fragile" if label == 1 else "Non-Fragile"}',
                          alpha=0.6, s=20, edgecolors='white', linewidth=0.5)
            
            dataset_display_names = {
                'train': 'Train',
                'val': 'Validation',  # Full name for validation
                'test': 'Test'
            }
            display_name = dataset_display_names.get(dataset_name, dataset_name.capitalize())
            ax.set_title(f'{display_name} Dataset', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{method}1')
            ax.set_ylabel(f'{method}2')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'HyenaDNA Model - {method} Embeddings Across Datasets', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        combined_path = os.path.join(save_dir, f'{method.lower()}_all_datasets.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} combined dataset plot to {combined_path}")
        plt.show()

def run_visualization_pipeline(model, train_loader, val_loader, test_loader, 
                              device, output_dir, max_samples=2000):
    
    print("\n" + "="*70)
    print("STARTING EMBEDDING VISUALIZATION PIPELINE - HyenaDNA Model")
    print("="*70)
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'embedding_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract embeddings from all datasets
    datasets = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    all_results = {}
    
    for dataset_name, dataloader in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Extract embeddings
        embeddings, labels, predictions = extract_embeddings_from_dataloader(
            model, dataloader, device, max_samples=max_samples
        )
        
        # Create dataset-specific directory
        dataset_viz_dir = os.path.join(viz_dir, dataset_name)
        os.makedirs(dataset_viz_dir, exist_ok=True)
        
        # Create visualizations
        reduced_embeddings = create_embedding_visualizations(
            embeddings, labels, predictions, 
            save_dir=dataset_viz_dir
        )
        
        
        all_results[dataset_name] = {
            'embeddings': embeddings,
            'labels': labels,
            'predictions': predictions,
            'reduced_embeddings': reduced_embeddings,
        }
        
        # Calculate classification accuracy for this dataset
        accuracy = np.mean(labels == predictions)
        print(f"\n{dataset_name.upper()} Dataset Accuracy: {accuracy:.4f}")
    
    # Create combined visualization with all datasets
    print("\nCreating combined dataset visualizations...")
    create_combined_dataset_plot(all_results, save_dir=viz_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION PIPELINE COMPLETE")
    print(f"All visualizations saved to: {viz_dir}")
    print("="*70)
    
    return all_results

class HyenaDNAModelWrapper:
    def __init__(self, model, tokenizer, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device
        self.model.eval()

    def _predict(self, sequences):
        all_probs = []
        batch_size = 16 

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            encoded = self.tokenizer(
                batch_seqs,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits')
                else:
                    logits = outputs
                
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[:, 1].cpu().numpy())
                
        return np.concatenate(all_probs)

    def _tokens_to_sequences(self, token_lists):
        sequences = []
        for tokens in token_lists:
            seq = ''.join([self.tokenizer.id_to_char.get(int(t), 'N') for t in tokens])
            sequences.append(seq)
        return sequences

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray) or (isinstance(inputs, list) and isinstance(inputs[0], (list, np.ndarray))):
            sequences = self._tokens_to_sequences(inputs)
        else:
            sequences = inputs if isinstance(inputs, list) else [inputs]

        return self._predict(sequences)

class ShapCompatibleWrapper:
    def __init__(self, model, tokenizer, device, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    def __call__(self, *args, **kwargs):
        # Handle various input formats from SHAP
        if len(args) > 0:
            input_data = args[0]
        else:
            input_data = kwargs.get('input_ids', kwargs.get('inputs', None))
        
        # Handle different input types
        if isinstance(input_data, str):
            # Single DNA sequence string
            encoded = self.tokenizer.encode(input_data, max_length=self.max_length)  # Use self.max_length
            input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long).to(self.device)
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str):
            # List of DNA sequence strings
            all_input_ids = []
            all_attention_masks = []
            for seq in input_data:
                encoded = self.tokenizer.encode(seq, max_length=self.max_length)  # Use self.max_length
                all_input_ids.append(encoded['input_ids'])
                all_attention_masks.append(encoded['attention_mask'])
            input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(all_attention_masks, dtype=torch.long).to(self.device)
        elif isinstance(input_data, np.ndarray):
            # Numpy array of token IDs
            input_ids = torch.from_numpy(input_data).long().to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], (int, float)):
            # List of token IDs
            input_ids = torch.tensor([input_data], dtype=torch.long).to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        elif isinstance(input_data, torch.Tensor):
            # Already a tensor
            input_ids = input_data.to(self.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Ensure correct shape
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Return probabilities for SHAP
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()


def create_shap_explainer_enhanced(model, tokenizer, background_sequences, explainer_type='partition', max_length=512):
    """Create an enhanced SHAP explainer with proper DNA masking"""
    print(f"Creating {explainer_type} SHAP explainer with DNA-specific masking...")

    # Create model wrapper first (needed for all explainer types)
    model_wrapper = HyenaDNAModelWrapper(model, tokenizer, max_length)

    if explainer_type == 'custom' or explainer_type == 'simple':
        # Custom DNA explainer with analyze_length attribute
        class SimpleDNAExplainer:
            def __init__(self, model, tokenizer, max_length):
                self.model = model
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.device = next(model.parameters()).device
                self.model.eval()
                self.window_size = 5
                self.analyze_length = min(200, max_length)  # Set analyze_length attribute

            def predict(self, sequences):
                if isinstance(sequences, str):
                    sequences = [sequences]

                encoded = tokenizer(
                    sequences,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=encoded['input_ids'].to(self.device),
                        attention_mask=encoded['attention_mask'].to(self.device)
                    )
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits')
                    else:
                        logits = outputs
                    probs = torch.softmax(logits, dim=1)
                    return probs[:, 1].cpu().numpy()

            def __call__(self, sequences):
                if isinstance(sequences, str):
                    sequences = [sequences]

                print(f"Analyzing {len(sequences)} sequences (first {self.analyze_length} nucleotides)...")
                results = []

                for seq_idx, seq in enumerate(sequences):
                    # Limit analysis to analyze_length
                    seq_to_analyze = seq[:self.analyze_length]

                    print(f"  Sequence {seq_idx+1}/{len(sequences)}")

                    # Get baseline
                    baseline = self.predict([seq_to_analyze])[0]

                    # Calculate importance using windows
                    importance = []
                    for i in range(0, len(seq_to_analyze), self.window_size):
                        masked = list(seq_to_analyze)
                        # Mask the window with 'N'
                        for j in range(i, min(i + self.window_size, len(seq_to_analyze))):
                            masked[j] = 'N'
                        masked_seq = ''.join(masked)

                        masked_pred = self.predict([masked_seq])[0]
                        window_imp = (baseline - masked_pred) / self.window_size

                        # Assign importance to each position in window
                        for j in range(i, min(i + self.window_size, len(seq_to_analyze))):
                            if j < len(importance):
                                importance[j] = window_imp
                            else:
                                importance.append(window_imp)

                    # Pad with zeros for the rest of the sequence
                    importance.extend([0] * (len(seq) - len(importance)))

                    result = SimpleNamespace(
                        values=np.array(importance[:len(seq)]),
                        base_values=baseline,
                        data=seq
                    )
                    results.append(result)

                print("Analysis complete!")
                return results

        explainer = SimpleDNAExplainer(model, tokenizer, max_length)
        explainer.analyze_length = explainer.analyze_length  # Ensure attribute is set
        return explainer

    elif explainer_type == 'text':
        # Text-based explainer with DNA masking
        def dna_model_wrapper(sequences):
            """Wrapper that handles DNA sequences"""
            if isinstance(sequences, str):
                sequences = [sequences]
            elif isinstance(sequences, np.ndarray):
                sequences = sequences.tolist()
            
            # Convert to strings if needed
            processed_seqs = []
            for seq in sequences:
                if isinstance(seq, (list, np.ndarray)) and not isinstance(seq, str):
                    # Convert token IDs back to sequence
                    seq_str = ''.join([tokenizer.id_to_char.get(int(t), 'N') 
                                      for t in seq if t != tokenizer.pad_token_id])
                    processed_seqs.append(seq_str)
                else:
                    processed_seqs.append(str(seq))
            
            return model_wrapper(processed_seqs)

        # Create baseline with N masking
        baseline_sequence = "N" * max_length
        
        # Create explainer with text masking
        explainer = shap.Explainer(
            dna_model_wrapper,
            masker=shap.maskers.Text(tokenizer=None, mask_token=baseline_sequence)
        )
        explainer.analyze_length = max_length
        return explainer
    

    elif explainer_type == 'partition':
        # Original partition explainer with analyze_length
        encoded = tokenizer(background_sequences[:20], max_length=max_length, 
                          padding='max_length', truncation=True)
        input_ids = np.array(encoded['input_ids'])
        masker = shap.maskers.Partition(input_ids)

        explainer = shap.Explainer(
            model_wrapper, # Pass the raw model instance
            masker=masker,

            algorithm='partition',
            max_evals=500,
            batch_size=1,
            silent=True
        )
        explainer.masker.data = input_ids
        explainer.masker._shape = input_ids.shape
        explainer.analyze_length = max_length
        return explainer

    elif explainer_type == 'permutation':
        # Windowed DNA explainer with analyze_length
        class WindowedDNAExplainer:
            def __init__(self, model, tokenizer, max_length, window_size=50):
                self.model = model
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.device = next(model.parameters()).device
                self.window_size = window_size
                self.model.eval()
                self.analyze_length = max_length  # Set analyze_length
            
            def __call__(self, inputs):
                if isinstance(inputs, np.ndarray) or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], (list, np.ndarray))):
                    sequences = []
                    for tokens in inputs:
                        seq = ''.join([
                            self.tokenizer.id_to_char.get(int(t), 'N')
                            for t in tokens if t != self.tokenizer.pad_token_id
                        ])
                        sequences.append(seq)
                else:
                    sequences = inputs if isinstance(inputs, list) else [inputs]
                
                results = []
                for seq in sequences:
                    encoded = self.tokenizer(
                        [seq],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=encoded['input_ids'].to(self.device),
                            attention_mask=encoded['attention_mask'].to(self.device)
                        )
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits')
                        else:
                            logits = outputs
                        baseline_prob = torch.softmax(logits, dim=1)[0, 1].cpu().item()
                    
                    importance = np.zeros(len(seq))
                    
                    for i in range(0, len(seq), self.window_size // 2):
                        end = min(i + self.window_size, len(seq))
                        masked_seq = list(seq)
                        for j in range(i, end):
                            masked_seq[j] = 'N'
                        masked_seq = ''.join(masked_seq)
                        
                        encoded = self.tokenizer(
                            [masked_seq],
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt"
                        )
                        
                        with torch.no_grad():
                            outputs = self.model(
                                input_ids=encoded['input_ids'].to(self.device),
                                attention_mask=encoded['attention_mask'].to(self.device)
                            )
                            if isinstance(outputs, dict):
                                logits = outputs.get('logits')
                            else:
                                logits = outputs
                            masked_prob = torch.softmax(logits, dim=1)[0, 1].cpu().item()
                        
                        window_importance = (baseline_prob - masked_prob) / (end - i)
                        for j in range(i, end):
                            importance[j] = window_importance
                    
                    result = SimpleNamespace(
                        values=importance,
                        base_values=baseline_prob,
                        data=seq
                    )
                    results.append(result)
                
                return results
        
        return WindowedDNAExplainer(model, tokenizer, max_length, window_size=50)

    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}. Use 'custom', 'simple', 'text', 'partition', or 'permutation'")

def create_shap_text_visualization(model, tokenizer, sequences, save_dir, max_length=512):
    """Create SHAP text visualization for DNA sequences"""
    print("Generating SHAP text visualization...")
    
    # Create a wrapper for the model
    def model_wrapper(seqs):
        if isinstance(seqs, np.ndarray):
            seqs = seqs.tolist()
        elif isinstance(seqs, str):
            seqs = [seqs]
        
        # Handle both raw sequences and tokenized inputs
        processed_seqs = []
        for seq in seqs:
            if isinstance(seq, (list, np.ndarray)) and not isinstance(seq, str):
                # Convert tokens back to sequence
                seq_str = ''.join([tokenizer.id_to_char.get(int(t), 'N') 
                                  for t in seq if t != tokenizer.pad_token_id])
                processed_seqs.append(seq_str)
            else:
                processed_seqs.append(str(seq))
        
        tokenized = tokenizer(
            processed_seqs,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        input_ids = tokenized["input_ids"].to(model.device if hasattr(model, 'device') else next(model.parameters()).device)
        attention_mask = tokenized["attention_mask"].to(input_ids.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
        return logits.cpu().numpy()

    # Create baseline sequence (all N's)
    baseline_sequence = "N" * max_length
    
    # Create SHAP explainer with text masker
    explainer = shap.Explainer(
        model_wrapper,
        masker=shap.maskers.Text(tokenizer=None, mask_token=baseline_sequence)
    )
    
    # Analyze sequences
    for i, seq in enumerate(sequences[:5]):  # Limit to first 5 for visualization
        print(f"Creating text visualization for sequence {i+1}...")
        
        # Truncate sequence if needed
        seq_truncated = seq[:max_length]
        
        # Get SHAP values
        shap_values = explainer([seq_truncated])
        
        # Create and save text plot
        shap_text_plot_path = os.path.join(save_dir, f"shap_text_visualization_{i+1}.png")
        shap.plots.text(shap_values, show=False)
        plt.savefig(shap_text_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {shap_text_plot_path}")
    
    return explainer

def analyze_sequence_with_shap(explainer, sequences, sequence_names=None):
    """
    Analyze sequences using SHAP explainer to get importance scores
    """
    print("Analyzing sequences with SHAP...")
    
    if isinstance(sequences, str):
        sequences = [sequences]
    
    try:
        # Check if it's a custom explainer or standard SHAP
        if hasattr(explainer, 'shap_values'):
            # Standard SHAP explainer
            shap_values = explainer.shap_values(sequences)
        else:
            # Custom explainer - call it directly
            shap_values = explainer(sequences)
        
        results = []
        for i, seq in enumerate(sequences):
            seq_name = sequence_names[i] if sequence_names else f"Sequence_{i}"
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list) and len(shap_values) > 0:
                # For lists of results from custom explainer
                if hasattr(shap_values[i], 'values'):
                    values = shap_values[i].values
                    base_value = shap_values[i].base_values if hasattr(shap_values[i], 'base_values') else 0
                else:
                    values = shap_values[i]
                    base_value = 0
            elif isinstance(shap_values, np.ndarray):
                # Standard SHAP output
                if len(shap_values.shape) > 1:
                    values = shap_values[i]
                else:
                    values = shap_values
                base_value = 0
            else:
                print(f"Unexpected SHAP values format: {type(shap_values)}")
                continue
            
            # Ensure values are numpy array
            if not isinstance(values, np.ndarray):
                values = np.array(values)
            
            # Ensure values match sequence length
            seq_length = len(seq)
            if len(values) > seq_length:
                values = values[:seq_length]
            elif len(values) < seq_length:
                values = np.pad(values, (0, seq_length - len(values)), constant_values=0)
            
            analysis = {
                'sequence_name': seq_name,
                'sequence': seq,
                'shap_values': values,
                'base_values': base_value,
                'values': values,
                'fragile_importance': values,
                'non_fragile_importance': -values
            }
            
            # Find top important positions
            if len(values) > 0:
                top_k = min(20, len(values))
                top_indices = np.argsort(np.abs(values))[-top_k:]
                analysis['top_important_positions'] = top_indices
                analysis['top_important_nucleotides'] = [seq[idx] for idx in top_indices if idx < len(seq)]
                analysis['top_importance_scores'] = values[top_indices]
            else:
                analysis['top_important_positions'] = []
                analysis['top_important_nucleotides'] = []
                analysis['top_importance_scores'] = []
            
            results.append(analysis)
        
        return results
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return []

def visualize_shap_results(shap_results, save_path="shap_analysis.png"):
    """
    Create comprehensive SHAP visualizations for fragile sites detection
    """
    print("Creating SHAP visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('SHAP Analysis Results for Fragile Sites Detection', fontsize=16)
    
    if len(shap_results) > 0:
        result = shap_results[0]
        
        # Nucleotide importance heatmap
        ax1 = axes[0, 0]
        importance_matrix = result['fragile_importance'].reshape(1, -1)
        seq_chars = list(result['sequence'][:len(result['fragile_importance'])])
        
        im = ax1.imshow(importance_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_title('Nucleotide Importance for Fragile Site Detection')
        ax1.set_xlabel('Nucleotide Position')
        ax1.set_ylabel('Sequence')
        
        if len(seq_chars) < 100:
            ax1.set_xticks(range(len(seq_chars)))
            ax1.set_xticklabels(seq_chars, rotation=90, fontsize=8)
        
        plt.colorbar(im, ax=ax1, label='SHAP Value')
        
        # Top important nucleotides
        ax2 = axes[0, 1]
        top_positions = result['top_important_positions'][-10:]
        top_scores = result['top_importance_scores'][-10:]
        
        bars = ax2.bar(range(len(top_scores)), top_scores, 
                      color=['red' if score > 0 else 'blue' for score in top_scores])
        ax2.set_title('Top 10 Most Important Nucleotides')
        ax2.set_xlabel('Nucleotide Position')
        ax2.set_ylabel('SHAP Value')
        ax2.set_xticks(range(len(top_scores)))
        ax2.set_xticklabels([f"{pos}\n({result['sequence'][pos] if pos < len(result['sequence']) else 'N'})" 
                            for pos in top_positions], rotation=45)
    
    # Distribution of SHAP values
    ax3 = axes[1, 0]
    all_shap_values = []
    for result in shap_results:
        all_shap_values.extend(result['fragile_importance'])
    
    ax3.hist(all_shap_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Distribution of SHAP Values (Fragile Class)')
    ax3.set_xlabel('SHAP Value')
    ax3.set_ylabel('Frequency')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Nucleotide composition analysis
    ax4 = axes[1, 1]
    nucleotide_importance = {'A': [], 'T': [], 'G': [], 'C': []}
    
    for result in shap_results:
        seq = result['sequence']
        importance = result['fragile_importance']
        
        for i, nucleotide in enumerate(seq):
            if i < len(importance) and nucleotide in nucleotide_importance:
                nucleotide_importance[nucleotide].append(importance[i])
    
    mean_importance = {}
    for nuc, values in nucleotide_importance.items():
        if values:
            mean_importance[nuc] = np.mean(values)
        else:
            mean_importance[nuc] = 0
    
    nucleotides = list(mean_importance.keys())
    importance_values = list(mean_importance.values())
    
    bars = ax4.bar(nucleotides, importance_values,
                  color=['red' if val > 0 else 'blue' for val in importance_values])
    ax4.set_title('Average SHAP Value by Nucleotide Type')
    ax4.set_xlabel('Nucleotide')
    ax4.set_ylabel('Average SHAP Value')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_shap_report(shap_results, model_performance, save_path="shap_report.txt"):
    """
    Generate comprehensive SHAP interpretation report for HyenaDNA fragile sites detection
    """
    print("Generating SHAP interpretation report...")
    
    report = []
    report.append("="*80)
    report.append("SHAP INTERPRETATION REPORT FOR HYENADNA FRAGILE SITES DETECTION")
    report.append("="*80)
    report.append("")
    
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append(f"  Test Accuracy: {model_performance.get('test_accuracy', 'N/A'):.4f}")
    report.append(f"  ROC-AUC Score: {model_performance.get('roc_auc', 'N/A'):.4f}")
    report.append(f"  Best Validation Accuracy: {model_performance.get('best_val_accuracy', 'N/A'):.4f}")
    report.append(f"  Best Validation Loss: {model_performance.get('best_val_loss', 'N/A'):.4f}")
    report.append("")
    
    report.append("SHAP ANALYSIS SUMMARY:")
    report.append(f"  Total sequences analyzed: {len(shap_results)}")
    report.append("")
    
    all_importance = []
    nucleotide_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    nucleotide_importance = {'A': [], 'T': [], 'G': [], 'C': []}
    
    for result in shap_results:
        seq = result['sequence']
        importance = result['fragile_importance']
        all_importance.extend(importance)
        
        for i, nuc in enumerate(seq):
            if i < len(importance) and nuc in nucleotide_counts:
                nucleotide_counts[nuc] += 1
                nucleotide_importance[nuc].append(importance[i])
    
    for result in shap_results:
        seq_length = len(result['sequence'])
        positive_shap = np.sum(result['fragile_importance'][result['fragile_importance'] > 0])
        # Normalize per kilobase (1000 bp)
        normalized_score = (positive_shap / seq_length) * 1000 if seq_length > 0 else 0
        result['normalized_score'] = normalized_score
        result['total_positive_shap'] = positive_shap
        
    report.append("TOP SEQUENCES WITH HIGHEST FRAGILE SITE SIGNALS (NORMALIZED PER KB):")
    
    # Sort by normalized score instead of total
    seq_scores = []
    for result in shap_results:
        seq_scores.append((
            result['sequence_name'], 
            result['total_positive_shap'],
            result['normalized_score'],  # Add normalized score
            result
        ))
    
    # Sort by normalized score (index 2) instead of total
    seq_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Update the display to show both scores
    for i, (seq_name, total_score, norm_score, result) in enumerate(seq_scores[:5]):
        report.append(f"  {i+1}. {seq_name}")
        report.append(f"     Total positive SHAP: {total_score:.4f}")
        report.append(f"     Normalized SHAP (per kb): {norm_score:.4f}")
        report.append(f"     Sequence length: {len(result['sequence'])}")
        
        importance = result['fragile_importance']
        top_indices = np.argsort(importance)[-5:]
        
        report.append("     Top contributing positions:")
        for idx in reversed(top_indices):
            if idx < len(result['sequence']):
                nuc = result['sequence'][idx]
                shap_val = importance[idx]
                report.append(f"       Position {idx}: {nuc} (SHAP: {shap_val:.4f})")
        report.append("")
    
    report.append("NUCLEOTIDE IMPORTANCE ANALYSIS:")
    for nuc in ['A', 'T', 'G', 'C']:
        if nucleotide_importance[nuc]:
            mean_imp = np.mean(nucleotide_importance[nuc])
            std_imp = np.std(nucleotide_importance[nuc])
            report.append(f"  {nuc}: Mean SHAP = {mean_imp:.4f} (±{std_imp:.4f})")
    report.append("")
    
    report.append("STATISTICAL INSIGHTS:")
    report.append(f"  Mean SHAP value: {np.mean(all_importance):.4f}")
    report.append(f"  Std SHAP value: {np.std(all_importance):.4f}")
    report.append(f"  % Positive contributions: {np.sum(np.array(all_importance) > 0) / len(all_importance) * 100:.1f}%")
    report.append(f"  % Negative contributions: {np.sum(np.array(all_importance) < 0) / len(all_importance) * 100:.1f}%")
    report.append("")
    
    report.append("BIOLOGICAL INTERPRETATION:")
    report.append("  • Positive SHAP values indicate nucleotides that increase fragile site probability")
    report.append("  • Negative SHAP values indicate nucleotides that decrease fragile site probability")
    report.append("  • High-importance regions may correspond to known fragile site motifs")
    report.append("  • Consider analyzing clustered high-importance positions for motif discovery")
    report.append("")
    
    report.append("="*80)
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"SHAP report saved to {save_path}")
    return '\n'.join(report)

def create_df_full_verified(pos_fasta, neg_fasta, verbose=False):
    """
    Verified version of create_df_full to ensure all sequences are loaded
    """
    if verbose:
        print("CREATING VERIFIED DATAFRAME")
    
    # Load positive sequences
    pos_data = []
    pos_count = 0
    try:
        for record in SeqIO.parse(pos_fasta, "fasta"):
            pos_data.append({
                'Name': record.id,
                'Sequence': str(record.seq),
                'Label': 1
            })
            pos_count += 1
        if verbose:
            print(f"Loaded {pos_count} positive sequences")
    except Exception as e:
        print(f"Error loading positive sequences: {e}")
        return None
    
    # Load negative sequences
    neg_data = []
    neg_count = 0
    try:
        for record in SeqIO.parse(neg_fasta, "fasta"):
            neg_data.append({
                'Name': record.id,
                'Sequence': str(record.seq),
                'Label': 0
            })
            neg_count += 1
        if verbose:
            print(f"Loaded {neg_count} negative sequences")
    except Exception as e:
        print(f"Error loading negative sequences: {e}")
        return None
    
    # Combine all data
    all_data = pos_data + neg_data
    df_full = pd.DataFrame(all_data)
    
    if verbose:
        print(f"Combined dataframe:")
        print(f"   Total: {len(df_full)} sequences")
        print(f"   Positive: {(df_full['Label'] == 1).sum()}")
        print(f"   Negative: {(df_full['Label'] == 0).sum()}")
    
    return df_full

def enhanced_balanced_split(df_full, test_size=0.2, val_size=0.1, random_state=42, 
                          balance_strategy='stratified', verbose=False):
    """
    Enhanced splitting with multiple balancing strategies
    """
    if verbose:
        print(f"\nCREATING ENHANCED BALANCED SPLITS (Strategy: {balance_strategy})")
    
    positive_count = (df_full['Label'] == 1).sum()
    negative_count = (df_full['Label'] == 0).sum()
    total_count = len(df_full)
    
    if verbose:
        print(f"Original class distribution:")
        print(f"   Positive: {positive_count} ({positive_count/total_count*100:.1f}%)")
        print(f"   Negative: {negative_count} ({negative_count/total_count*100:.1f}%)")
    
    if balance_strategy == 'stratified':
        train_df, temp_df = train_test_split(
            df_full, test_size=(test_size + val_size), 
            stratify=df_full['Label'], random_state=random_state
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(test_size / (test_size + val_size)),
            stratify=temp_df['Label'], random_state=random_state
        )
        
    elif balance_strategy == 'undersample':
        min_class_size = min(positive_count, negative_count)
        
        positive_samples = df_full[df_full['Label'] == 1].sample(
            n=min_class_size, random_state=random_state
        )
        negative_samples = df_full[df_full['Label'] == 0].sample(
            n=min_class_size, random_state=random_state
        )
        
        balanced_df = pd.concat([positive_samples, negative_samples], 
                               ignore_index=True).sample(frac=1, random_state=random_state)
        
        train_df, temp_df = train_test_split(
            balanced_df, test_size=(test_size + val_size),
            stratify=balanced_df['Label'], random_state=random_state
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(test_size / (test_size + val_size)),
            stratify=temp_df['Label'], random_state=random_state
        )
        
    elif balance_strategy == 'oversample':
        max_class_size = max(positive_count, negative_count)
        
        if positive_count < negative_count:
            positive_samples = df_full[df_full['Label'] == 1].sample(
                n=max_class_size, replace=True, random_state=random_state
            )
            negative_samples = df_full[df_full['Label'] == 0]
        else:
            positive_samples = df_full[df_full['Label'] == 1]
            negative_samples = df_full[df_full['Label'] == 0].sample(
                n=max_class_size, replace=True, random_state=random_state
            )
        
        balanced_df = pd.concat([positive_samples, negative_samples], 
                               ignore_index=True).sample(frac=1, random_state=random_state)
        
        train_df, temp_df = train_test_split(
            balanced_df, test_size=(test_size + val_size),
            stratify=balanced_df['Label'], random_state=random_state
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(test_size / (test_size + val_size)),
            stratify=temp_df['Label'], random_state=random_state
        )
        
    elif balance_strategy == 'combined':
        min_test_size = min(
            int(positive_count * test_size),
            int(negative_count * test_size)
        )
        
        if min_test_size == 0:
            min_test_size = min(positive_count, negative_count) // 2
        
        test_positive = df_full[df_full['Label'] == 1].sample(
            n=min_test_size, random_state=random_state
        )
        test_negative = df_full[df_full['Label'] == 0].sample(
            n=min_test_size, random_state=random_state
        )
        
        test_df = pd.concat([test_positive, test_negative], ignore_index=True)
        
        remaining_indices = df_full.index.difference(test_df.index)
        remaining_df = df_full.loc[remaining_indices].reset_index(drop=True)
        
        adjusted_val_size = val_size / (1 - test_size)
        
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=adjusted_val_size,
            stratify=remaining_df['Label'],
            random_state=random_state
        )
        
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}. " +
                        f"Use 'stratified', 'undersample', 'oversample', or 'combined'")
    
    # Reset indices for all dataframes
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Print final statistics
    if verbose:
        print(f"Final split statistics:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            pos_count = (split_df['Label'] == 1).sum()
            neg_count = (split_df['Label'] == 0).sum()
            total = len(split_df)
            pos_pct = pos_count / total * 100 if total > 0 else 0
            balance_ratio = pos_count / neg_count if neg_count > 0 else float('inf')
            
            print(f"   {split_name}: {total} total")
            print(f"     Positive: {pos_count} ({pos_pct:.1f}%)")
            print(f"     Negative: {neg_count} ({100-pos_pct:.1f}%)")
            print(f"     Ratio (pos:neg): {balance_ratio:.2f}:1")
    
    return train_df, val_df, test_df

def find_fragile_site_motifs(shap_results, motif_length=600):
    """
    Find sequence motifs associated with high SHAP values
    """
    motif_importance = {}
    
    if not shap_results:
        print("No SHAP results available for motif analysis")
        return []
    
    for result in shap_results:
        seq = result['sequence']
        importance = result['fragile_importance']
        
        # Skip if sequence is too short
        if len(seq) < motif_length:
            continue
            
        # Skip if importance values are invalid
        if len(importance) == 0 or np.all(np.isnan(importance)):
            continue
        
        # Find high-importance regions (top 5%)
        threshold = np.percentile(importance[~np.isnan(importance)], 95)
        
        for i in range(len(seq) - motif_length + 1):
            if i + motif_length <= len(importance):
                window_importance = importance[i:i+motif_length]
                if np.mean(window_importance) > threshold:
                    motif = seq[i:i+motif_length]
                    avg_importance = np.mean(window_importance)
                    
                    if motif not in motif_importance:
                        motif_importance[motif] = []
                    motif_importance[motif].append(avg_importance)
    
    # Average importance per motif (only keep motifs seen at least twice)
    motif_scores = {
        motif: np.mean(scores)
        for motif, scores in motif_importance.items()
        if len(scores) >= 2
    }
    
    sorted_motifs = sorted(motif_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_motifs:
        print(f"\nFound {len(sorted_motifs)} recurring motifs")
        print("\nTop 10 Fragile Site Motifs:")
        for i, (motif, score) in enumerate(sorted_motifs[:10]):
            if len(motif) > 40:
                display_motif = f"{motif[:20]}...{motif[-20:]}"
            else:
                display_motif = motif
            print(f"  {i+1}. {display_motif} (length: {len(motif)}bp): {score:.4f}")
    else:
        print("No recurring motifs found")
    
    return sorted_motifs

def create_simple_dataset(df, tokenizer, max_length=1024, batch_size=16, shuffle=True, num_workers=2):
    """Create dataset compatible with our model"""
    sequences = df['Sequence'].tolist()
    labels = df['Label'].tolist()

    all_input_ids = []
    all_attention_masks = []

    print(f"Tokenizing {len(sequences)} sequences...")
    for seq in tqdm(sequences):
        encoding = tokenizer.encode(seq, max_length=max_length)
        all_input_ids.append(encoding['input_ids'])
        all_attention_masks.append(encoding['attention_mask'])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_attention_masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataset, dataloader

def create_dna_shap_visualization(sequence, shap_values, true_label, pred_label, 
                                  save_path, seq_idx=1):
    
    # Extract SHAP values from different formats
    if hasattr(shap_values, 'values'):
        if hasattr(shap_values.values, 'shape') and len(shap_values.values.shape) > 1:
            values = shap_values.values[0] if shap_values.values.shape[0] > 0 else shap_values.values
        else:
            values = shap_values.values
    elif isinstance(shap_values, list):
        values = shap_values[0] if len(shap_values) > 0 else []
    else:
        values = shap_values
    
    # Ensure numpy array
    values = np.array(values).flatten()
    
    # Truncate to sequence length
    seq_len = min(len(sequence), len(values))
    sequence = sequence[:seq_len]
    values = values[:seq_len]
    
    # Create figure with specific size
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # Normalize SHAP values for coloring
    vmin, vmax = values.min(), values.max()
    # Ensure symmetric colormap around 0
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Create color map
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r  # Red for positive, Blue for negative
    
    # Plot each nucleotide with colored background
    for i, (nucleotide, shap_val) in enumerate(zip(sequence, values)):
        color = cmap(norm(shap_val))
        
        # Create text with colored background
        ax.text(i, 0, nucleotide, 
                fontsize=10, 
                ha='center', 
                va='center',
                family='monospace',
                weight='bold' if abs(shap_val) > abs_max * 0.5 else 'normal',
                bbox=dict(boxstyle="square,pad=0.3", 
                         facecolor=color, 
                         edgecolor='black',
                         linewidth=0.5,
                         alpha=0.9))
    
    # Set plot properties
    ax.set_xlim(-1, len(sequence))
    ax.set_ylim(-0.5, 0.5)
    
    # Title with true and predicted labels
    true_label_text = "Fragile" if true_label == 1 else "Non-Fragile"
    pred_label_text = "Fragile" if pred_label == 1 else "Non-Fragile"
    ax.set_title(f'DNA SHAP Analysis - Sequence {seq_idx}\n'
                 f'True: {true_label_text}, Predicted: {pred_label_text}',
                 fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Nucleotide Position', fontsize=12)
    ax.set_yticks([])
    
    # Remove y-axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add gridlines for position reference
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                       shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Importance Score', fontsize=11)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=1e-6):
    """Create cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_dna_shap_visualization(sequence, shap_values, true_label, pred_label, 
                                  save_path, seq_idx=1):
    # Extract SHAP values from different formats
    if hasattr(shap_values, 'values'):
        if hasattr(shap_values.values, 'shape') and len(shap_values.values.shape) > 1:
            values = shap_values.values[0] if shap_values.values.shape[0] > 0 else shap_values.values
        else:
            values = shap_values.values
    elif isinstance(shap_values, list):
        values = shap_values[0] if len(shap_values) > 0 else []
    else:
        values = shap_values
    
    # Ensure numpy array
    values = np.array(values).flatten()
    
    # Truncate to sequence length
    seq_len = min(len(sequence), len(values))
    sequence = sequence[:seq_len]
    values = values[:seq_len]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    
    # Normalize SHAP values for coloring
    vmin, vmax = values.min(), values.max()
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Create color map
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r
    
    # Plot each nucleotide with colored background
    for i, (nucleotide, shap_val) in enumerate(zip(sequence, values)):
        color = cmap(norm(shap_val))
        
        ax.text(i, 0, nucleotide, 
                fontsize=10, 
                ha='center', 
                va='center',
                family='monospace',
                weight='bold' if abs(shap_val) > abs_max * 0.5 else 'normal',
                bbox=dict(boxstyle="square,pad=0.3", 
                         facecolor=color, 
                         edgecolor='black',
                         linewidth=0.5,
                         alpha=0.9))
    
    # Set plot properties
    ax.set_xlim(-1, len(sequence))
    ax.set_ylim(-0.5, 0.5)
    
    # Title with labels
    true_label_text = "Fragile" if true_label == 1 else "Non-Fragile"
    pred_label_text = "Fragile" if pred_label == 1 else "Non-Fragile"
    ax.set_title(f'DNA SHAP Analysis - Sequence {seq_idx}\n'
                 f'True: {true_label_text}, Predicted: {pred_label_text}',
                 fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Nucleotide Position', fontsize=12)
    ax.set_yticks([])
    
    # Remove y-axis spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add gridlines
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', 
                       shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Importance Score', fontsize=11)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Saved DNA visualization: {os.path.basename(save_path)}")

def create_multi_sequence_shap_visualization(sequences, shap_values_list, 
                                            true_labels, pred_labels, 
                                            save_path, max_sequences=5):
    
    num_sequences = min(len(sequences), max_sequences)
    fig, axes = plt.subplots(num_sequences, 1, figsize=(20, 4*num_sequences))
    
    if num_sequences == 1:
        axes = [axes]
    
    for seq_idx in range(num_sequences):
        ax = axes[seq_idx]
        sequence = sequences[seq_idx]
        shap_values = shap_values_list[seq_idx]
        true_label = true_labels[seq_idx]
        pred_label = pred_labels[seq_idx]
        
        # Process SHAP values
        if hasattr(shap_values, 'values'):
            values = shap_values.values
        else:
            values = shap_values
        values = np.array(values).flatten()
        
        # Limit sequence length
        seq_len = min(len(sequence), len(values), 256)
        sequence = sequence[:seq_len]
        values = values[:seq_len]
        
        # Color normalization
        abs_max = max(abs(values.min()), abs(values.max()))
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = plt.cm.RdBu_r
        
        # Plot nucleotides
        for i, (nuc, val) in enumerate(zip(sequence, values)):
            color = cmap(norm(val))
            ax.text(i, 0, nuc, 
                   fontsize=8,
                   ha='center', 
                   va='center',
                   family='monospace',
                   bbox=dict(boxstyle="square,pad=0.2", 
                            facecolor=color, 
                            edgecolor='gray',
                            linewidth=0.3,
                            alpha=0.9))
        
        # Formatting
        ax.set_xlim(-1, len(sequence))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        # Labels
        true_text = "Fragile" if true_label == 1 else "Non-Fragile"
        pred_text = "Fragile" if pred_label == 1 else "Non-Fragile"
        
        if seq_idx == 0:
            ax.set_title(f'DNA SHAP Analysis - Multiple Sequences\n'
                        f'Seq {seq_idx+1}: True={true_text}, Pred={pred_text}',
                        fontsize=12)
        else:
            ax.set_title(f'Seq {seq_idx+1}: True={true_text}, Pred={pred_text}',
                        fontsize=10)
        
        if seq_idx == num_sequences - 1:
            ax.set_xlabel('Nucleotide Position', fontsize=10)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('SHAP Importance Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig


def create_shap_html_visualization(model, tokenizer, sequence, output_dir, max_length=256):
    """
    Create a SINGLE SHAP HTML visualization file for HyenaDNA model
    Fixed version that properly handles HyenaDNA tokenization
    """
    import shap
    import numpy as np
    import torch
    import os
    
    print("\n" + "="*60)
    print("CREATING SHAP HTML VISUALIZATION (HyenaDNA)")
    print("="*60)
    
    # Create a simple wrapper that works with sequences directly
    class DNAModelWrapper:
        def __init__(self, model, tokenizer, max_length, device):
            self.model = model
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device
            self.model.eval()
        
        def __call__(self, sequences):
            # Handle different input types
            if isinstance(sequences, str):
                sequences = [sequences]
            elif isinstance(sequences, np.ndarray):
                if sequences.ndim == 1:
                    sequences = sequences.tolist()
                else:
                    sequences = [s for s in sequences]
            
            # Process each sequence
            all_probs = []
            for seq in sequences:
                # Convert to string if needed
                if isinstance(seq, (list, np.ndarray)):
                    seq = ''.join(str(s) for s in seq)
                
                # Tokenize
                encoding = self.tokenizer.encode(seq, max_length=self.max_length)
                input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(self.device)
                attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(self.device)
                
                # Get probabilities
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits')
                    else:
                        logits = outputs
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    all_probs.append(probs[0].cpu().numpy())
            
            return np.array(all_probs)
    
    # Setup device
    device = next(model.parameters()).device

    
    # Create model wrapper
    model_wrapper = ShapCompatibleWrapper(model, tokenizer, device, max_length=max_length)
    
    # Truncate sequence if too long
    if len(sequence) > 500:
        sequence_to_analyze = sequence[:500]
        print(f"Sequence truncated to 500bp for visualization")
    else:
        sequence_to_analyze = sequence
    
    print(f"Analyzing sequence of length {len(sequence_to_analyze)}...")
    
    # Use a simpler approach - window-based importance calculation
    print("Computing importance scores using window-based method...")
    
    # Get baseline prediction
    baseline_probs = model_wrapper([sequence_to_analyze])[0]
    baseline_score = baseline_probs[1]  # Fragile class probability
    
    # Calculate importance for each position using masking
    importance_scores = []
    window_size = 5
    
    for i in range(0, len(sequence_to_analyze), window_size):
        # Create masked sequence
        masked_seq = list(sequence_to_analyze)
        for j in range(i, min(i + window_size, len(sequence_to_analyze))):
            masked_seq[j] = 'N'
        masked_seq = ''.join(masked_seq)
        
        # Get prediction for masked sequence
        masked_probs = model_wrapper([masked_seq])[0]
        masked_score = masked_probs[1]
        
        # Calculate importance (difference from baseline)
        importance = baseline_score - masked_score
        
        # Assign to each position in the window
        for j in range(i, min(i + window_size, len(sequence_to_analyze))):
            if j < len(sequence_to_analyze):
                importance_scores.append(importance)
    
    # Ensure we have scores for all positions
    importance_scores = importance_scores[:len(sequence_to_analyze)]
    
    # Create the HTML visualization
    html_path = os.path.join(output_dir, "hyena_shap_dna_visualization.html")
    
    try:
        # Try to use SHAP's built-in visualization if possible
        # Create a simple Explanation object
        shap_values = shap.Explanation(
            values=np.array([importance_scores]),
            base_values=np.array([baseline_score]),
            data=np.array([list(sequence_to_analyze)]),
            feature_names=list(sequence_to_analyze)
        )
        
        # Try to create SHAP plot
        try:
            html_output = shap.plots.text(shap_values, display=False)
            
            if isinstance(html_output, str):
                html_string = html_output
            elif hasattr(html_output, 'data'):
                html_string = html_output.data
            else:
                html_string = str(html_output)
            
            with open(html_path, 'w') as f:
                f.write(html_string)
            
            print(f"✓ SHAP text visualization saved to: hyena_shap_dna_visualization.html")
            return True
            
        except Exception as e:
            print(f"Could not create SHAP text plot: {e}")
            raise  # Re-raise to trigger fallback
            
    except Exception as e:
        print(f"Creating custom HTML visualization instead...")
        
        # Create custom HTML visualization
        create_custom_hyena_html_simple(
            sequence_to_analyze, 
            importance_scores, 
            baseline_score,
            html_path, 
            model_name="HyenaDNA"
        )
        print(f"✓ Custom HTML visualization saved to: hyena_shap_dna_visualization.html")
        return True

def plot_roc_auc_curve(y_true, y_scores, save_path="roc_auc_curve.png", model_name="HyenaDNA"):
    """
    Plot ROC curve with AUC score for model predictions.
    Modified for clearer audience understanding.
    """
    print(f"\nGenerating 'How well does our model work?' plot for {model_name}...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot with enhanced styling
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve with gradient effect
    plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
             label=f'Our Model (AUC = {roc_auc:.3f})', alpha=0.8) # Changed legend label
    
    # Add confidence interval shading (simulated)
    plt.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Guessing') # Simplified label
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Incorrect Fragile Site Predictions (False Positive Rate)', fontsize=14, fontweight='bold') # Clarified label
    plt.ylabel('Correct Fragile Site Predictions (True Positive Rate)', fontsize=14, fontweight='bold') # Clarified label
    plt.title(f'How well does the {model_name} Model predict fragile sites?', 
              fontsize=16, fontweight='bold', pad=20) # New, more direct title
    
    # Add legend with custom styling
    legend = plt.legend(loc="lower right", fontsize=12, frameon=True, 
                       fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)
    
    # Add AUC score text box
    textstr = f'AUC Score: {roc_auc:.4f}\nModel: {model_name}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.text(0.6, 0.15, textstr, fontsize=12, verticalalignment='top', 
             bbox=props, fontweight='bold')
    
    # Add optimal threshold point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', s=100, 
               zorder=5, label=f'Best Threshold: {optimal_threshold:.3f}') # Simplified label
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"ROC curve saved to: {save_path}")
    print(f"AUC Score: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    return roc_auc, optimal_threshold

def plot_fragile_sites_heatmap(shap_results, save_path="fragile_sites_heatmap.png", 
                               max_sequences=20, max_length=500):
    """
    Create a heatmap visualization of fragile site SHAP values across multiple sequences.
    Modified for clearer audience understanding.
    """
    print(f"\nGenerating 'Fragile vs. Protective' heatmap...")
    
    if not shap_results:
        print("No SHAP results available for heatmap")
        return
    
    # Prepare data for heatmap
    sequences_to_plot = min(len(shap_results), max_sequences)
    
    # Create matrix for heatmap
    heatmap_data = []
    sequence_names = []
    
    for i in range(sequences_to_plot):
        result = shap_results[i]
        importance = result['fragile_importance'][:max_length]
        
        # Pad if shorter than max_length
        if len(importance) < max_length:
            importance = np.pad(importance, (0, max_length - len(importance)), 
                              constant_values=0)
        
        heatmap_data.append(importance)
        # Simplify the sequence names for better readability
        name_parts = result['sequence_name'].split('_')
        simplified_name = f"{name_parts[0]}_{name_parts[-1]}" 
        sequence_names.append(simplified_name)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main heatmap
    im = ax1.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', 
                   interpolation='nearest')
    
    # Set labels
    ax1.set_xlabel('Position in DNA Sequence', fontsize=14, fontweight='bold') # Clarified label
    ax1.set_ylabel('DNA Sequences', fontsize=14, fontweight='bold') # Simplified label
    ax1.set_title('Which DNA locations increase or decrease fragility?', 
                 fontsize=16, fontweight='bold', pad=20) # New, more direct title
    
    # Set y-axis labels
    ax1.set_yticks(range(len(sequence_names)))
    ax1.set_yticklabels(sequence_names, fontsize=10)
    
    # Set x-axis labels (show every 50th position)
    x_positions = range(0, max_length, 50)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_positions, rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', 
                       pad=0.1, fraction=0.05)
    cbar.set_label('Fragile-Promoting (Red) vs. Protective (Blue)', 
                  fontsize=12, fontweight='bold') # Simplified label
    
    # Add average importance plot below
    avg_importance = np.mean(heatmap_data, axis=0)
    ax2.plot(avg_importance, color='#2c3e50', linewidth=2, alpha=0.8)
    ax2.fill_between(range(len(avg_importance)), avg_importance, 
                    where=(avg_importance > 0), color='#e74c3c', alpha=0.3, 
                    label='Promotes Fragility') # Simplified label
    ax2.fill_between(range(len(avg_importance)), avg_importance, 
                    where=(avg_importance < 0), color='#3498db', alpha=0.3, 
                    label='Protects Against Fragility') # Simplified label
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Position in DNA Sequence', fontsize=14, fontweight='bold') # Clarified label
    ax2.set_ylabel('Average Fragility Impact', fontsize=12, fontweight='bold') # Simplified label
    ax2.set_title('Average Fragility Impact Across All Sequences', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Set x-axis for average plot
    ax2.set_xlim(0, max_length)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_positions, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved to: {save_path}")
    
    # Print statistics
    print(f"Heatmap Statistics:")
    print(f"  - Sequences displayed: {sequences_to_plot}")
    print(f"  - Sequence length: {max_length} bp")
    print(f"  - Max Fragility Impact: {np.max(heatmap_data):.4f}")
    print(f"  - Min Fragility Impact: {np.min(heatmap_data):.4f}")
    print(f"  - Mean Fragility Impact: {np.mean(heatmap_data):.4f}")

def plot_fragile_sites_curves(shap_results, save_path="fragile_sites_curves.png", 
                            num_examples=5):
    """
    Plot SHAP values as curves for easy interpretation of fragile site patterns.
    Modified for clearer audience understanding.
    """
    print(f"\nGenerating 'Fragile vs. Protective' curve visualizations...")
    
    if not shap_results:
        print("No SHAP results available for curves")
        return
    
    # Select sequences with highest variability in SHAP values for interesting plots
    sequences_with_variance = []
    for result in shap_results[:min(len(shap_results), 20)]:
        variance = np.var(result['fragile_importance'])
        sequences_with_variance.append((result, variance))
    
    # Sort by variance and take top examples
    sequences_with_variance.sort(key=lambda x: x[1], reverse=True)
    selected_results = [x[0] for x in sequences_with_variance[:num_examples]]
    
    # Create subplots
    fig, axes = plt.subplots(num_examples, 1, figsize=(16, 3*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for idx, result in enumerate(selected_results):
        ax = axes[idx]
        
        importance = result['fragile_importance']
        positions = np.arange(len(importance))
        
        # Smooth the curve for better visualization
        smoothed_importance = gaussian_filter1d(importance, sigma=2)
        
        # Plot the curve
        ax.plot(positions, importance, alpha=0.3, color='gray', 
               linewidth=0.5, label='Original Impact Score') # Simplified label
        ax.plot(positions, smoothed_importance, color='#2c3e50', 
               linewidth=2, label='Smoothed Impact Score') # Simplified label
        
        # Fill areas
        ax.fill_between(positions, 0, smoothed_importance, 
                       where=(smoothed_importance > 0), 
                       color='#e74c3c', alpha=0.4, label='Promotes Fragility') # Simplified label
        ax.fill_between(positions, 0, smoothed_importance, 
                       where=(smoothed_importance < 0), 
                       color='#3498db', alpha=0.4, label='Protects Against Fragility') # Simplified label
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Mark regions of high importance
        threshold = np.percentile(np.abs(importance), 90)
        high_importance = np.abs(importance) > threshold
        
        # Highlight high importance regions
        for i in range(len(high_importance)):
            if high_importance[i]:
                ax.axvspan(i-1, i+1, alpha=0.1, color='yellow')
        
        # Labels and formatting
        # Simplify sequence name
        name_parts = result['sequence_name'].split('_')
        simplified_name = f"Sequence: {name_parts[0]}_{name_parts[-1]}"
        ax.set_title(simplified_name, fontsize=12, fontweight='bold') # Simplified title
        ax.set_xlabel('Position in DNA Sequence' if idx == len(selected_results)-1 else '', 
                     fontsize=11) # Clarified label
        ax.set_ylabel('Fragility Impact Score', fontsize=11) # Simplified label
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Add legend only to first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
        
        # Add statistics text
        stats_text = f'Max Impact: {np.max(importance):.3f}, Min Impact: {np.min(importance):.3f}, Average Impact: {np.mean(importance):.3f}' # Simplified text
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle('How DNA Locations Affect Fragile Site Predictions', 
                fontsize=16, fontweight='bold', y=1.02) # New, more direct title
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Curve plots saved to: {save_path}")
    print(f"Plotted {len(selected_results)} sequences with highest SHAP variance")


def create_custom_hyena_html_simple(sequence, importance_scores, baseline_score, output_path, model_name="HyenaDNA"):
    import numpy as np
    import os
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy array
    values = np.array(importance_scores)
    
    # Normalize values
    max_val = np.max(np.abs(values)) if len(values) > 0 and not np.all(np.isnan(values)) else 1
    if max_val == 0:
        max_val = 1
    
    # Calculate statistics 
    avg_importance = np.mean(np.abs(values))
    num_promoting = np.sum(values > 0)
    num_protective = np.sum(values < 0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP DNA Analysis - {model_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 20px;
                background: linear-gradient(135deg, #9b59b6 0%, #e74c3c 100%);
            }}
            .container {{
                background-color: white;
                padding: 35px;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 1400px;
                margin: 0 auto;
            }}
            .model-badge {{
                display: inline-block;
                background: linear-gradient(135deg, #9b59b6, #e74c3c);
                color: white;
                padding: 10px 25px;
                border-radius: 25px;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 30px;
            }}
            .info-box {{
                background: linear-gradient(to right, #f8f9fa, #e9ecef);
                border-left: 5px solid #9b59b6;
                padding: 20px;
                margin: 25px 0;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .class-labels {{
                display: flex;
                justify-content: center;
                gap: 60px;
                margin: 25px 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .output-0 {{ 
                color: #3498db;
                padding: 12px 25px;
                border: 3px solid #3498db;
                border-radius: 10px;
                background: rgba(52, 152, 219, 0.08);
                transition: all 0.3s;
            }}
            .output-1 {{ 
                color: #e74c3c;
                padding: 12px 25px;
                border: 3px solid #e74c3c;
                border-radius: 10px;
                background: rgba(231, 76, 60, 0.08);
                transition: all 0.3s;
            }}
            .output-0:hover, .output-1:hover {{
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .sequence-container {{
                font-family: 'Courier New', monospace;
                font-size: 16px;
                line-height: 2.5;
                word-wrap: break-word;
                padding: 30px;
                background: linear-gradient(145deg, #ffffff, #f5f5f5);
                border: 2px solid #dee2e6;
                border-radius: 12px;
                margin: 25px 0;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            }}
            .nucleotide {{
                display: inline-block;
                padding: 4px 6px;
                margin: 2px;
                border-radius: 4px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-shadow: 0 1px 3px rgba(0,0,0,0.2);
                border: 1px solid rgba(0,0,0,0.1);
            }}
            .nucleotide:hover {{
                transform: scale(1.4) translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                z-index: 100;
                position: relative;
            }}
            .legend {{
                margin-top: 35px;
                padding: 25px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 12px;
                box-shadow: 0 3px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #9b59b6;
            }}
            .legend h3 {{
                margin-top: 0;
                color: #2c3e50;
                font-size: 20px;
                margin-bottom: 15px;
            }}
            .legend ul {{
                margin: 15px 0;
                padding-left: 25px;
            }}
            .legend li {{
                margin: 8px 0;
                color: #34495e;
            }}
            .legend p {{
                color: #555;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="model-badge">{model_name} Model</div>
            <h1>ðŸ§¬ DNA Sequence SHAP Analysis</h1>
            <p class="subtitle">Analyzing nucleotide importance for fragile site prediction</p>
            
            <div class="info-box">
                <strong>ðŸ"Š Analysis Summary:</strong><br>
                â€¢ Sequence Length: {len(sequence)} bp<br>
                â€¢ Baseline Fragile Probability: {baseline_score:.4f}<br>
                â€¢ Importance Range: [{np.min(values):.4f}, {np.max(values):.4f}]
            </div>
            
            <div class="class-labels">
                <span class="output-0">Class 0: Non-Fragile Site</span>
                <span class="output-1">Class 1: Fragile Site</span>
            </div>
            
            <div class="sequence-container">
    """
    
    # Add each nucleotide with coloring 
    for i, (nucleotide, val) in enumerate(zip(sequence[:len(values)], values)):
        # Calculate color intensity
        intensity = min(abs(val) / max_val, 1.0) if max_val > 0 else 0
        
        if val > 0:  # Promotes fragile
            r = 255
            g = int(255 * (1 - intensity * 0.8))
            b = int(255 * (1 - intensity * 0.8))
        else:  # Promotes non-fragile
            r = int(255 * (1 - intensity * 0.8))
            g = int(255 * (1 - intensity * 0.8))
            b = 255
        
        # Text color based on background intensity
        text_color = 'white' if intensity > 0.6 else 'black'
        
        # Create style
        style = f"background-color: rgb({r},{g},{b}); color: {text_color};"
        
        html_content += f'<span class="nucleotide" style="{style}" '
        html_content += f'title="Position {i+1} | {nucleotide} | Score: {val:.4f}">{nucleotide}</span>'
        
        if (i + 1) % 60 == 0:
            html_content += '<br>'
    
    html_content += f"""
            </div>
            
            <div class="legend">
                <h3>ðŸ"Š Interpretation Guide - {model_name} Model</h3>
                <p><strong>Understanding the Visualization:</strong></p>
                <ul style="line-height: 1.8;">
                    <li>ðŸ"´ <strong>Red nucleotides:</strong> Increase fragile site probability (risk factors)</li>
                    <li>ðŸ"µ <strong>Blue nucleotides:</strong> Decrease fragile site probability (protective factors)</li>
                    <li>âšª <strong>White/light colors:</strong> Minimal impact on prediction</li>
                    <li>ðŸ"ˆ <strong>Color intensity:</strong> Stronger colors = higher importance scores</li>
                </ul>
                <p style="margin-top: 15px;">
                    <strong>Model Statistics:</strong><br>
                    â€¢ Average Absolute Importance: {avg_importance:.4f}<br>
                    â€¢ Importance Ratio (Risk/Protective): {num_promoting}/{num_protective}<br>
                    â€¢ Model Architecture: {model_name} Transformer
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("HYENADNA TRAINING WITH SHAP ANALYSIS FOR FRAGILE SITE DETECTION")
    print("="*70)

    # Set random seed for reproducibility
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- STEP 1: DATA PREPARATION ---
    print("\n" + "="*70)
    print("DATA LOADING AND PREPARATION")
    print("="*70)
    
    df_full = create_df_full_verified(args.pos_fasta, args.neg_fasta, verbose=args.verbose)
    if df_full is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)

    print(f"\nDATASET LOADED SUCCESSFULLY")
    print(f"Total samples: {len(df_full)}")
    print(f"Positive samples: {(df_full['Label'] == 1).sum()}")
    print(f"Negative samples: {(df_full['Label'] == 0).sum()}")

    # Split data
    train_df, val_df, test_df = enhanced_balanced_split(
        df_full,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        balance_strategy=args.balance_strategy,
        verbose=args.verbose
    )

    # Calculate class weights
    train_labels = train_df['Label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\nCLASS WEIGHTS CALCULATED:")
    print(f"   Negative (0): {class_weight_dict[0]:.3f}")
    print(f"   Positive (1): {class_weight_dict[1]:.3f}")

    # --- STEP 2: MODEL & TOKENIZER INITIALIZATION ---
    print("\n" + "="*70)
    print("MODEL AND TOKENIZER INITIALIZATION")
    print("="*70)

    model_config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'd_inner': args.d_inner,
        'order': args.order,
        'seq_len': args.seq_len,
        'num_classes': 2,
        'dropout': args.dropout,
        'prenorm': True
    }

    tokenizer = HyenaDNATokenizer()
    model = HyenaDNAModel(**model_config)
    
    if args.bidirectional:
        print("Using Bidirectional HyenaDNA model...")
        model = BidirectionalHyenaWrapper(
            hyena_model=model,
            num_classes=2,
            bidirectional_strategy=args.bidirectional_strategy
        ).to(device)
        print(f"Using Bidirectional HyenaDNA with {args.bidirectional_strategy} strategy")
    else:
        print("Using Standard (Unidirectional) HyenaDNA model...")
        model = model.to(device)
    
    print(f"Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    print("\nCREATING DATASETS...")
    train_dataset, train_dataloader = create_simple_dataset(
        train_df, tokenizer,
        max_length=args.seq_len,
        batch_size=args.batch_size,
        num_workers=2
    )
    val_dataset, val_dataloader = create_simple_dataset(
        val_df, tokenizer,
        max_length=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    test_dataset, test_dataloader = create_simple_dataset(
        test_df, tokenizer,
        max_length=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Initialize scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        num_training_steps
    )

    # Initialize loss function
    class_weight_tensor = torch.tensor(
        [class_weight_dict[0], class_weight_dict[1]],
        dtype=torch.float32
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    # --- STEP 3: TRAINING WITH RESOURCE MONITORING ---
    print("\n" + "="*70)
    print("STARTING TRAINING WITH RESOURCE MONITORING")
    print("="*70)

    # Initialize resource monitoring
    training_resource_stats = {
        'epoch_times': [],
        'gpu_memory_used': [],
        'gpu_memory_total': [],
        'cpu_percent': [],
        'ram_percent': [],
        'ram_used_gb': []
    }

    # Training tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    training_history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'test_losses': [],
        'test_accuracies': [],
        'learning_rates': []
    }

    print(f"TRAINING CONFIGURATION:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max epochs: {args.num_epochs}")
    print(f"   Early stopping patience: {args.patience}")

    total_training_start = time.time()

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        
        # Record resources at epoch start
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        ram_percent = memory_info.percent
        ram_used_gb = memory_info.used / (1024**3)
        
        # GPU monitoring
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available() and device.type == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            input_ids, attention_mask, labels = [x.to(device, non_blocking=True) for x in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
            
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

            # Update progress bar with resource info
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_train_predictions/total_train_samples:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'GPU': f'{gpu_memory_used:.1f}GB'
                })

        train_loss /= len(train_dataset)
        train_accuracy = correct_train_predictions / total_train_samples
        training_history['train_losses'].append(train_loss)
        training_history['train_accuracies'].append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs} [Val]'):
                input_ids, attention_mask, labels = [x.to(device, non_blocking=True) for x in batch]

                outputs = model(input_ids, attention_mask)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits')
                else:
                    logits = outputs
                
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        val_loss /= len(val_dataset)
        val_accuracy = correct_val_predictions / total_val_samples
        training_history['val_losses'].append(val_loss)
        training_history['val_accuracies'].append(val_accuracy)

        # Test evaluation
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids, attention_mask, labels = [x.to(device, non_blocking=True) for x in batch]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                loss = loss_fn(logits, labels)

                test_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                correct_test_predictions += (predicted == labels).sum().item()
                total_test_samples += labels.size(0)

        test_loss /= len(test_dataset)
        test_accuracy = correct_test_predictions / total_test_samples
        training_history['test_losses'].append(test_loss)
        training_history['test_accuracies'].append(test_accuracy)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Record epoch metrics
        epoch_duration = time.time() - epoch_start_time
        training_resource_stats['epoch_times'].append(epoch_duration)
        training_resource_stats['gpu_memory_used'].append(gpu_memory_used)
        training_resource_stats['gpu_memory_total'].append(gpu_memory_total)
        training_resource_stats['cpu_percent'].append(cpu_percent)
        training_resource_stats['ram_percent'].append(ram_percent)
        training_resource_stats['ram_used_gb'].append(ram_used_gb)

        # Print epoch summary with resource info
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'  Resources - Time: {epoch_duration:.2f}s | GPU: {gpu_memory_used:.2f}/{gpu_memory_total:.2f}GB | RAM: {ram_percent:.1f}%')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f'  New best validation loss: {best_val_loss:.4f}')

            # Save best model
            model_save_path = os.path.join(args.output_dir, args.model_save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'model_config': model_config,
                'args': vars(args)
            }, model_save_path)
        else:
            epochs_without_improvement += 1
            print(f'  No improvement for {epochs_without_improvement} epochs')

        if epochs_without_improvement >= args.patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break

        print('-' * 50)

    total_training_time = time.time() - total_training_start

    # Save training resource report
    training_resource_path = os.path.join(args.output_dir, 'training_resource_report.txt')
    with open(training_resource_path, 'w') as f:
        f.write("HYENADNA TRAINING RESOURCE USAGE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        if training_resource_stats['epoch_times']:
            f.write(f"Average Epoch Time: {np.mean(training_resource_stats['epoch_times']):.2f} seconds\n")
        if training_resource_stats['gpu_memory_used']:
            f.write(f"Max GPU Memory Used: {max(training_resource_stats['gpu_memory_used']):.2f} GB\n")
            f.write(f"Average GPU Memory Used: {np.mean(training_resource_stats['gpu_memory_used']):.2f} GB\n")
        else:
            f.write("GPU Memory: Not available\n")
        if training_resource_stats['cpu_percent']:
            f.write(f"Peak CPU Usage: {max(training_resource_stats['cpu_percent']):.1f}%\n")
            f.write(f"Average CPU Usage: {np.mean(training_resource_stats['cpu_percent']):.1f}%\n")
        else:
            f.write("CPU Usage: Not available\n")
        if training_resource_stats['ram_percent']:
            f.write(f"Peak RAM Usage: {max(training_resource_stats['ram_percent']):.1f}% ({max(training_resource_stats['ram_used_gb']):.2f} GB)\n")
            f.write(f"Average RAM Usage: {np.mean(training_resource_stats['ram_percent']):.1f}% ({np.mean(training_resource_stats['ram_used_gb']):.2f} GB)\n")
        else:
            f.write("RAM Usage: Not available\n")

    print(f"Training resource report saved to: {training_resource_path}")

    # --- STEP 4: MODEL EVALUATION ---
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")

    # Final evaluation
    model.eval()
    final_test_loss = 0.0
    correct_final_predictions = 0
    total_final_samples = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final evaluation"):
            input_ids, attention_mask, labels = [x.to(device, non_blocking=True) for x in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            loss = loss_fn(logits, labels)
            probabilities = torch.softmax(logits, dim=1)

            final_test_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct_final_predictions += (predicted == labels).sum().item()
            total_final_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)

    print("\n" + "="*70)
    print("GENERATING ROC/AUC CURVE")
    print("="*70)

# Extract positive class probabilities
    positive_class_probs = all_probabilities[:, 1]

# Plot ROC curve
    roc_save_path = os.path.join(args.output_dir, 'roc_auc_curve.png')
    auc_score, optimal_threshold = plot_roc_auc_curve(
        all_labels, 
        positive_class_probs,
        save_path=roc_save_path,
        model_name="HyenaDNA"
)

    final_test_loss /= len(test_dataset)
    final_test_accuracy = correct_final_predictions / total_final_samples

    # Save training history
    training_history.update({
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'final_test_loss': final_test_loss,
        'final_test_accuracy': final_test_accuracy,
        'args': vars(args)
    })

    history_save_path = os.path.join(args.output_dir, args.history_save_name)
    torch.save(training_history, history_save_path)
    print(f"Training history saved as '{history_save_path}'")

    # Print final results
    print(f'\nFINAL RESULTS:')
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    print(f'Best Validation Accuracy: {best_val_accuracy:.4f}')
    print(f'Final Test Loss: {final_test_loss:.4f}')
    print(f'Final Test Accuracy: {final_test_accuracy:.4f}')

    # Detailed metrics
    print("\n=== DETAILED PERFORMANCE REPORT ===")
    print(classification_report(all_labels, all_predictions,
                              target_names=['Non-Fragile', 'Fragile']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fragile', 'Fragile'],
                yticklabels=['Non-Fragile', 'Fragile'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ROC-AUC Score
    auc_score = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # --- STEP 5: EMBEDDING VISUALIZATION (Optional) ---
    if args.enable_visualization:
        print("\n" + "="*70)
        print("EMBEDDING VISUALIZATION")
        print("="*70)
        visualization_results = run_visualization_pipeline(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            args.output_dir,
            max_samples=args.viz_max_samples
        )
        print("\nVisualization complete!")
        print(f"Check {args.output_dir}/embedding_visualizations/ for results")

    # --- STEP 6: SHAP ANALYSIS ---
    if args.enable_shap:
        print("\n" + "="*70)
        print("SHAP INTERPRETATION ANALYSIS")
        print("="*70)
        
        try:
            # Create model performance dictionary
            model_performance = {
                'test_accuracy': final_test_accuracy,
                'roc_auc': auc_score,
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss': best_val_loss,
                'optimal_threshold': optimal_threshold
            }
            
            # Prepare sequences for SHAP analysis
            test_sequences = test_df['Sequence'].tolist()
            test_names = test_df['Name'].tolist()
            
            # HTML Visualization
            print("\n" + "="*70)
            print("CREATING HTML VISUALIZATION")
            print("="*70)
            
            html_sequence = test_sequences[0] if test_sequences else None
            
            if html_sequence:
                html_success = create_shap_html_visualization(
                    model=model,
                    tokenizer=tokenizer,
                    sequence=html_sequence,
                    output_dir=args.output_dir,
                    max_length=256
                )
                
                if html_success:
                    print("\n✓ HTML visualization created successfully!")
                    print(f"Files created:")
                    print(f"  • HyenaDNA: {args.output_dir}/hyena_shap_dna_visualization.html")
            
            # Create SHAP explainer
            print("\n--- Creating Enhanced SHAP Explainer ---")
            background_sequences = test_sequences[:args.shap_background_size]
            explainer = create_shap_explainer_enhanced(
                model,
                tokenizer,
                background_sequences,
                args.shap_explainer,
                args.seq_len
            )
            
            # Select analysis indices
            analysis_indices = []
            tp_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions))
                        if true == 1 and pred == 1]
            tn_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions))
                        if true == 0 and pred == 0]
            fp_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions))
                        if true == 0 and pred == 1]
            fn_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions))
                        if true == 1 and pred == 0]
            
            analysis_indices.extend(tp_indices[:args.shap_tp_samples])
            analysis_indices.extend(tn_indices[:args.shap_tn_samples])
            analysis_indices.extend(fp_indices[:args.shap_fp_samples])
            analysis_indices.extend(fn_indices[:args.shap_fn_samples])
            analysis_indices = analysis_indices[:args.shap_analysis_size]
            
            # Get sequences for analysis
            shap_sequences = [test_sequences[i] for i in analysis_indices]
            shap_names = [f"{test_names[i]}_TrueLabel{all_labels[i]}_Pred{all_predictions[i]}"
                        for i in analysis_indices]
            
            # Perform SHAP analysis
            shap_results = analyze_sequence_with_shap(
                explainer,
                shap_sequences,
                shap_names
            )

            print("\n" + "="*70)
            print("GENERATING FRAGILE SITES HEATMAP")
            print("="*70)

            heatmap_save_path = os.path.join(args.output_dir, 'fragile_sites_heatmap.png')
            plot_fragile_sites_heatmap(
                shap_results,
                save_path=heatmap_save_path,
                max_sequences=20,
                max_length=500
            )

            # ===== NEW: Generate Fragile Sites Curves =====
            print("\n" + "="*70)
            print("GENERATING FRAGILE SITES CURVES")
            print("="*70)

            curves_save_path = os.path.join(args.output_dir, 'fragile_sites_curves.png')
            plot_fragile_sites_curves(
                shap_results,
                save_path=curves_save_path,
                num_examples=5
            )
            
            # Create visualizations
            shap_plot_path = os.path.join(args.output_dir, args.shap_plot_name)
            visualization_figure = visualize_shap_results(
                shap_results,
                save_path=shap_plot_path
            )
            
            # Generate report
            shap_report_path = os.path.join(args.output_dir, args.shap_report_name)
            generate_shap_report(
                shap_results,
                model_performance,
                save_path=shap_report_path
            )
            
            # Find most important sequences
            seq_scores = []
            for result in shap_results:
                positive_shap = np.sum(result['fragile_importance'][result['fragile_importance'] > 0])
                seq_scores.append((result['sequence_name'], positive_shap))
            
            seq_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTOP 5 SEQUENCES WITH HIGHEST FRAGILE SITE SIGNALS:")
            for i, (seq_name, score) in enumerate(seq_scores[:5]):
                print(f"   {i+1}. {seq_name}: {score:.4f}")
            
            # === MOTIF DISCOVERY FROM SHAP SIGNALS ===
            motifs = find_fragile_site_motifs(shap_results, motif_length=600)
            
            # Save top motifs to a file
            motif_output_path = os.path.join(args.output_dir, "fragile_site_motifs.txt")
            with open(motif_output_path, "w") as f:
                f.write("Top Fragile Site Motifs (from SHAP signals):\n")
                f.write("="*60 + "\n")
                for i, (motif, score) in enumerate(motifs[:10]):
                    f.write(f"{i+1}. {motif}: {score:.4f}\n")
            
            print(f"Motif results saved to {motif_output_path}")
            
            # Save additional SHAP data
            shap_data = {
                'shap_results': shap_results,
                'model_performance': model_performance,
                'analysis_indices': analysis_indices,
                #'nucleotide_importance': nucleotide_importance,
                'args': vars(args)
            }
            
            shap_data_path = os.path.join(args.output_dir, args.shap_data_name)
            torch.save(shap_data, shap_data_path)
            
            print(f"\nSHAP analysis data saved to '{shap_data_path}'")
            
            print("\n" + "="*70)
            print("SHAP ANALYSIS COMPLETE")
            print("Generated files:")
            print(f"   • {shap_plot_path} (visualizations)")
            print(f"   • {shap_report_path} (detailed report)")
            print(f"   • {heatmap_save_path} (fragile sites heatmap)")  
            print(f"   • {curves_save_path} (fragile sites curves)")     
            print(f"   • {roc_save_path} (ROC/AUC curve)")             
            print(f"   • {shap_data_path} (analysis data)")
            print("="*70)
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            print("Continuing without SHAP analysis...")
    
    else:
        print("\nSHAP analysis disabled.")
    
    print(f"\nTraining and analysis complete!")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Final ROC-AUC: {auc_score:.4f}")

if __name__ == "__main__":
    main()