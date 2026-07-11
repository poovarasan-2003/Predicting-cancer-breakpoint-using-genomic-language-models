import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import argparse
import umap
import os
import time
import psutil
from transformers import PreTrainedTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional, Dict
from Bio import SeqIO
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')



class BidirectionalCNN(nn.Module):
    def __init__(self, num_filters=128, filter_sizes=[3, 7, 15, 31, 63], 
                 num_classes=2, dropout=0.5, input_channels=4,
                 bidirectional_strategy='concat'):
        super(BidirectionalCNN, self).__init__()
        
        self.bidirectional_strategy = bidirectional_strategy
        
        # Forward stream
        self.forward_convs = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=fs, 
                     padding=fs//2) for fs in filter_sizes
        ])
        
        # Reverse stream (separate parameters)
        self.reverse_convs = nn.ModuleList([
            nn.Conv1d(input_channels, num_filters, kernel_size=fs, 
                     padding=fs//2) for fs in filter_sizes
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Adjust FC layer size based on strategy
        if bidirectional_strategy == 'concat':
            fc_input_size = num_filters * len(filter_sizes) * 2
        else:
            fc_input_size = num_filters * len(filter_sizes)
            
        self.fc = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        
        # Forward processing
        forward_features = []
        for conv in self.forward_convs:
            h = self.relu(conv(x))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            forward_features.append(h)
        
        # Reverse processing
        x_reversed = torch.flip(x, dims=[2])
        reverse_features = []
        for conv in self.reverse_convs:
            h = self.relu(conv(x_reversed))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            reverse_features.append(h)
        
        # Combine based on strategy
        if self.bidirectional_strategy == 'concat':
            combined = torch.cat(forward_features + reverse_features, dim=1)
        elif self.bidirectional_strategy == 'add':
            forward_combined = torch.cat(forward_features, dim=1)
            reverse_combined = torch.cat(reverse_features, dim=1)
            combined = forward_combined + reverse_combined
        elif self.bidirectional_strategy == 'max':
            forward_combined = torch.cat(forward_features, dim=1)
            reverse_combined = torch.cat(reverse_features, dim=1)
            combined = torch.max(forward_combined, reverse_combined)
        else:
            raise ValueError(f"Unknown strategy: {self.bidirectional_strategy}")
        
        out = self.dropout(combined)
        return self.fc(out)
        
    

# DATA LOADING AND PREPROCESSING FUNCTIONS
def load_and_create_dataset(pos_fasta_path, neg_fasta_path):
    """Load sequences from FASTA files and create combined dataset"""
    print("Loading FASTA data...")
    
    # Load positive sequences
    pos_sequences = []
    pos_names = []
    for record in SeqIO.parse(pos_fasta_path, "fasta"):
        pos_sequences.append(str(record.seq).upper())
        pos_names.append(record.id)
    
    # Load negative sequences  
    neg_sequences = []
    neg_names = []
    for record in SeqIO.parse(neg_fasta_path, "fasta"):
        neg_sequences.append(str(record.seq).upper())
        neg_names.append(record.id)
    
    # Create combined dataset
    sequences = pos_sequences + neg_sequences
    labels = [1] * len(pos_sequences) + [0] * len(neg_sequences)
    names = pos_names + neg_names
    
    print(f"Loaded {len(pos_sequences)} positive sequences")
    print(f"Loaded {len(neg_sequences)} negative sequences")
    print(f"Total: {len(sequences)} sequences")
    
    return sequences, labels, names

def balance_and_split_data(sequences, labels, names, balance_strategy='combined', random_state=42):
    """Balance dataset and create train/val/test splits - memory efficient version"""
    print(f"Balancing dataset using '{balance_strategy}' strategy...")
    
    # Get original class distribution
    y = np.array(labels)
    unique, counts = np.unique(y, return_counts=True)
    print(f"Original distribution: {dict(zip(unique, counts))}")
    
    # Memory-efficient balancing using indices instead of full sequences
    indices = np.arange(len(sequences))
    
    if balance_strategy == 'undersample':
        # Manual undersampling - much more memory efficient
        pos_indices = indices[y == 1]
        neg_indices = indices[y == 0]
        
        # Sample to match the smaller class
        min_count = min(len(pos_indices), len(neg_indices))
        np.random.seed(random_state)
        
        selected_pos = np.random.choice(pos_indices, min_count, replace=False)
        selected_neg = np.random.choice(neg_indices, min_count, replace=False)
        
        balanced_indices = np.concatenate([selected_pos, selected_neg])
        
    elif balance_strategy == 'oversample':
        # Manual oversampling
        pos_indices = indices[y == 1]
        neg_indices = indices[y == 0]
        
        # Sample to match the larger class
        max_count = max(len(pos_indices), len(neg_indices))
        np.random.seed(random_state)
        
        selected_pos = np.random.choice(pos_indices, max_count, replace=True)
        selected_neg = np.random.choice(neg_indices, max_count, replace=True)
        
        balanced_indices = np.concatenate([selected_pos, selected_neg])
        
    elif balance_strategy == 'combined':
        # Simple combined approach - undersample majority, slightly oversample minority
        pos_indices = indices[y == 1]
        neg_indices = indices[y == 0]
        
        if len(pos_indices) > len(neg_indices):
            # More positives than negatives
            target_count = int(len(neg_indices) * 1.2)  # Slightly more than minority
            np.random.seed(random_state)
            selected_pos = np.random.choice(
                pos_indices, 
                target_count, 
                replace=(target_count > len(pos_indices))
            )
            selected_neg = np.random.choice(
                neg_indices, 
                target_count, 
                replace=(target_count > len(neg_indices))
            )

        else:
            # More negatives than positives
            target_count = int(len(pos_indices) * 1.2)  # Slightly more than minority
            np.random.seed(random_state)
            selected_pos = np.random.choice(
                pos_indices, 
                target_count, 
                replace=(target_count > len(pos_indices))
            )
            selected_neg = np.random.choice(
                neg_indices, 
                target_count, 
                replace=(target_count > len(neg_indices))
            )
 
        
        balanced_indices = np.concatenate([selected_pos, selected_neg])
        
    elif balance_strategy == 'stratified':
        # Keep original distribution
        balanced_indices = indices
    else:
        raise ValueError(f"Unknown balance strategy: {balance_strategy}")
    
    # Apply balancing using indices
    sequences_balanced = [sequences[i] for i in balanced_indices]
    labels_balanced = [labels[i] for i in balanced_indices]
    names_balanced = [names[i] if i < len(names) else f"seq_{i}" for i in balanced_indices]
    
    # Get new class distribution
    y_balanced = np.array(labels_balanced)
    unique, counts = np.unique(y_balanced, return_counts=True)
    print(f"Balanced distribution: {dict(zip(unique, counts))}")
    
    # Create train/val/test splits
    print("Creating data splits...")
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, names_trainval, names_test = train_test_split(
        sequences_balanced, labels_balanced, names_balanced, 
        test_size=0.2, random_state=random_state, stratify=labels_balanced
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
        X_trainval, y_trainval, names_trainval, 
        test_size=0.2, random_state=random_state, stratify=y_trainval
    )
    
    print(f"Train: {len(X_train)} sequences")
    print(f"Val: {len(X_val)} sequences")  
    print(f"Test: {len(X_test)} sequences")
    
    # Print class distributions for each split
    for split_name, (seqs, lbls) in [("Train", (X_train, y_train)), ("Val", (X_val, y_val)), ("Test", (X_test, y_test))]:
        pos_count = sum(lbls)
        neg_count = len(lbls) - pos_count
        total = len(lbls)
        pos_pct = pos_count / total * 100 if total > 0 else 0
        balance_ratio = pos_count / neg_count if neg_count > 0 else float('inf')
        
        print(f"   {split_name}: {total} total")
        print(f"     Positive: {pos_count} ({pos_pct:.1f}%)")
        print(f"     Negative: {neg_count} ({100-pos_pct:.1f}%)")
        print(f"     Ratio (pos:neg): {balance_ratio:.2f}:1")
    
    return (X_train, y_train, names_train), (X_val, y_val, names_val), (X_test, y_test, names_test)

# CNN MODEL ARCHITECTURE

class CNN(nn.Module):

    def __init__(self, num_filters=128, filter_sizes=[3, 7, 15, 31, 63], 
                 num_classes=2, dropout=0.5, vocab_size=12, embedding_dim=100):
        super(CNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, 
                     padding=fs//2) for fs in filter_sizes
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Store the embedding dimension for visualization
        self.embedding_size = num_filters * len(filter_sizes)
        self.fc = nn.Linear(self.embedding_size, num_classes)
        
    def extract_embeddings(self, x):
        x = self.embedding(x)  
        x = x.transpose(1, 2)  
        
        features = []
        for conv in self.convs:
            h = self.relu(conv(x))
            h = F.max_pool1d(h, h.size(2)).squeeze(2)
            features.append(h)
        
        # Combined features - these are our embeddings
        combined = torch.cat(features, dim=1)
        return combined
    
    def forward(self, x):
        """Standard forward pass for classification"""
        embeddings = self.extract_embeddings(x)
        out = self.dropout(embeddings)
        return self.fc(out)

# TOKENIZER CLASS 

class CNNDNATokenizer(PreTrainedTokenizer):
    def __init__(self, model_max_length=6144, **kwargs):
        # Character mapping
        self.char_to_id = {
            '[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[BOS]': 3, 
            '[MASK]': 4, '[RESERVED]': 5, '[UNK]': 6,
            'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 11
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        super().__init__(
            pad_token="[PAD]",
            unk_token="[UNK]", 
            mask_token="[MASK]",
            model_max_length=model_max_length,
            **kwargs
        )
    
    def get_vocab(self):
        """Required method for HuggingFace tokenizer"""
        return self.char_to_id.copy()
    
    @property
    def vocab_size(self):
        return len(self.char_to_id)
    
    @property
    def pad_token_id(self):
        return self.char_to_id['[PAD]']
    
    @property
    def unk_token_id(self):
        return self.char_to_id['[UNK]']
    
    @property
    def mask_token_id(self):
        return self.char_to_id['[MASK]']
    
    def _tokenize(self, text: str):
        """Tokenize a string into individual characters"""
        return list(text.upper())
    
    def _convert_token_to_id(self, token: str):
        """Convert a token to its ID"""
        return self.char_to_id.get(token, self.char_to_id['[UNK]'])
    
    def _convert_id_to_token(self, index: int):
        """Convert an ID to its token"""
        return self.id_to_char.get(index, '[UNK]')
    
    def convert_tokens_to_string(self, tokens):
        """Convert tokens back to string"""
        return "".join(tokens)
    
    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        """Main tokenization method with offset_mapping support"""
        # Check if it's a single string (which is the common case for your dataset)
        single_input = isinstance(text, str)
        
        if single_input:
            text = [text]
        
        result = {'input_ids': [], 'attention_mask': []}
        if return_offsets_mapping:
            result['offset_mapping'] = []
        
        for sequence in text:
            # Tokenize
            tokens = self._tokenize(sequence)
            token_ids = [self._convert_token_to_id(token) for token in tokens]
            
            # Handle max length
            max_length = kwargs.get('max_length', self.model_max_length)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                tokens = tokens[:max_length]
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Padding
            if kwargs.get('padding') == 'max_length':
                pad_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            result['input_ids'].append(token_ids)
            result['attention_mask'].append(attention_mask)
            
            # Add offset mapping if requested
            if return_offsets_mapping:
                offset_mapping = [(i, i+1) for i in range(len(tokens))]
                if kwargs.get('padding') == 'max_length':
                    offset_mapping.extend([(0, 0)] * pad_length)
                result['offset_mapping'].append(offset_mapping)
        
        # If single input, return the first (and only) element directly
        if single_input:
            result['input_ids'] = result['input_ids'][0]
            result['attention_mask'] = result['attention_mask'][0]
            if return_offsets_mapping:
                result['offset_mapping'] = result['offset_mapping'][0]
        
        # Convert to tensors if requested
        if kwargs.get('return_tensors') == 'pt':
            import torch
            for key in result:
                result[key] = torch.tensor(result[key])
        
        return result

# DATASET CLASS

def create_simple_dataset(df, tokenizer, max_length=6144, batch_size=4, shuffle=True, num_workers=2):
    sequences = df['Sequence'].tolist() if 'Sequence' in df.columns else df
    labels = df['Label'].tolist() if 'Label' in df.columns else [0] * len(sequences)
    
    dataset = SimpleDNADataset(sequences, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataset, dataloader

class SimpleDNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=6144):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        encoded = self.tokenizer(sequence, max_length=self.max_length, 
                                padding='max_length', truncation=True)
        input_tensor = torch.tensor(encoded['input_ids'], dtype=torch.long)

        return {
            'input_ids': input_tensor,
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
def extract_embeddings_from_dataloader(model, dataloader, device, max_samples=None):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get embeddings
            embeddings = model.extract_embeddings(input_ids)
            
            # Get predictions
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            
            sample_count += len(labels)
            if max_samples and sample_count >= max_samples:
                break
    
    # Concatenate all batches
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
    
    plt.suptitle('CNN Embeddings Visualization: DNA Fragile Sites Detection', 
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
        
        plt.suptitle(f'{method} Visualization - CNN DNA Fragile Sites Model', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(save_dir, f'{method.lower()}_visualization.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} visualization to {individual_path}")
        plt.show()
    
    return reduced_embeddings

# SHAP MODEL WRAPPER

class CNNModelWrapper:
    """Wrapper for CNN model to work with SHAP """
    def __init__(self, model, tokenizer, max_length=6144):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device
        self.model.eval()
        
    def __call__(self, sequences):
        if isinstance(sequences, np.ndarray):
            # Already tokenized (SHAP input)
            input_ids = torch.tensor(sequences, dtype=torch.long).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities[:, 1].cpu().numpy()
        
        elif isinstance(sequences, str):
            sequences = [sequences]
        
        # Tokenize if not already
        encoded = self.tokenizer(sequences, max_length=self.max_length,
                                padding='max_length', truncation=True, return_tensors="pt")
        
        input_ids = encoded['input_ids'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities[:, 1].cpu().numpy()

def create_shap_html_visualization(model, tokenizer, sequence, output_dir, max_length=256):
    """
    Create a SINGLE SHAP HTML visualization file for CNN model
    Exactly like the Caduceus implementation - one file output
    """
    print("\n" + "="*60)
    print("CREATING SHAP HTML VISUALIZATION")
    print("="*60)
    
    # Create a probability wrapper for SHAP text visualization
    class ProbabilityWrapper:
        def __init__(self, model, tokenizer, max_length, device):
            self.model = model
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device
            self.model.eval()
        
        def __call__(self, sequences):
            # Handle different input types
            if isinstance(sequences, np.ndarray):
                sequences = sequences.tolist()
            if isinstance(sequences, str):
                sequences = [sequences]
            
            # Tokenize
            encoded = self.tokenizer(
                sequences, 
                max_length=self.max_length,
                padding='max_length', 
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            
            # Get probabilities (IMPORTANT: must return probabilities, not logits!)
            with torch.no_grad():
                outputs = self.model(input_ids)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
            
            return probs.cpu().numpy()
    
    # Setup device
    device = next(model.parameters()).device
    
    # Create probability wrapper
    prob_wrapper = ProbabilityWrapper(model, tokenizer, max_length, device)
    
    # Create text masker for DNA sequences
    masker = shap.maskers.Text(tokenizer, mask_token="N")
    
    # Create SHAP explainer with text masker
    explainer = shap.Explainer(
        prob_wrapper,
        masker,
        output_names=['Non-Fragile', 'Fragile']
    )
    
    # Truncate sequence if too long
    if len(sequence) > 500:
        sequence_to_analyze = sequence[:500]
        print(f"Sequence truncated to 500bp for visualization")
    else:
        sequence_to_analyze = sequence
    
    print(f"Analyzing sequence of length {len(sequence_to_analyze)}...")
    print("Computing SHAP values...")
    
    # Compute SHAP values
    shap_values = explainer([sequence_to_analyze])
    
    # Create the HTML visualization - SINGLE FILE
    html_path = os.path.join(output_dir, "shap_dna_visualization.html")
    
    print("Creating text visualization...")
    
    try:
        # Get HTML output from SHAP
        html_output = shap.plots.text(shap_values, display=False)
        
        # Handle different return types
        if isinstance(html_output, str):
            html_string = html_output
        elif hasattr(html_output, 'data'):
            html_string = html_output.data
        else:
            html_string = str(html_output)
        
        # Save to file
        with open(html_path, 'w') as f:
            f.write(html_string)
        
        print(f"✓ SHAP text visualization saved to: shap_dna_visualization.html")
        return True
        
    except Exception as e:
        print(f"⚠ Could not create SHAP text plot: {e}")
        print("Creating custom HTML visualization instead...")
        
        # Fallback: Create custom HTML visualization
        create_custom_dna_html(shap_values, sequence_to_analyze, html_path)
        print(f"✓ Custom HTML visualization saved to: shap_dna_visualization.html")
        return True

def create_custom_dna_html(shap_values, sequence, output_path):
    """
    Create custom HTML visualization as fallback - SINGLE FILE
    """
    
    # Extract SHAP values
    if hasattr(shap_values, 'values'):
        values = shap_values.values[0]
        if len(values.shape) > 1:
            values = values[:, 1]  # Fragile class
    else:
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
    
    # Normalize values
    max_val = np.max(np.abs(values)) if len(values) > 0 and not np.all(np.isnan(values)) else 1
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP DNA Sequence Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .class-labels {
                display: flex;
                justify-content: center;
                gap: 50px;
                margin: 20px 0;
                font-size: 18px;
                font-weight: bold;
            }
            .output-0 { 
                color: #0066cc;
                padding: 10px 20px;
                border: 2px solid #0066cc;
                border-radius: 5px;
            }
            .output-1 { 
                color: #cc0000;
                padding: 10px 20px;
                border: 2px solid #cc0000;
                border-radius: 5px;
            }
            .sequence-container {
                font-family: 'Courier New', monospace;
                font-size: 16px;
                line-height: 2.2;
                word-wrap: break-word;
                padding: 25px;
                background-color: #fafafa;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
            }
            .nucleotide {
                display: inline-block;
                padding: 3px 5px;
                margin: 1px;
                border-radius: 3px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s;
            }
            .nucleotide:hover {
                transform: scale(1.2);
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .legend {
                margin-top: 30px;
                padding: 20px;
                background-color: #f0f0f0;
                border-radius: 8px;
            }
            .legend h3 {
                margin-top: 0;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-top: 20px;
                text-align: center;
            }
            .stat-box {
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
            .stat-label {
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 SHAP Analysis - DNA Sequence Interpretation</h1>
            <div class="class-labels">
                <div class="output-0">Output 0: Non-Fragile Site</div>
                <div class="output-1">Output 1: Fragile Site</div>
            </div>
    """
    
    # Add statistics
    num_promoting = np.sum(values > 0)
    num_protective = np.sum(values < 0)
    avg_importance = np.mean(np.abs(values))
    
    html_content += f"""
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{len(sequence)}</div>
                    <div class="stat-label">Sequence Length (bp)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_promoting}</div>
                    <div class="stat-label">Fragile-Promoting Positions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_protective}</div>
                    <div class="stat-label">Protective Positions</div>
                </div>
            </div>
            <div class="sequence-container">
    """
    
    # Add each nucleotide with coloring
    for i, (nucleotide, val) in enumerate(zip(sequence[:len(values)], values)):
        # Calculate color
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
        
        # Add nucleotide span with tooltip
        html_content += f'<span class="nucleotide" style="{style}" title="Position {i+1} | {nucleotide} | SHAP: {val:.4f}">{nucleotide}</span>'
        
        # Line break every 60 characters for readability
        if (i + 1) % 60 == 0:
            html_content += '<br>'
    
    html_content += f"""
            </div>
            <div class="legend">
                <h3>📊 Interpretation Guide</h3>
                <p><strong>Color Intensity:</strong> Stronger colors indicate higher SHAP values (more important positions)</p>
                <p>
                    <span style="background-color: #ffcccc; padding: 5px; border-radius: 3px;">Light Red</span> → 
                    <span style="background-color: #ff6666; padding: 5px; border-radius: 3px;">Medium Red</span> → 
                    <span style="background-color: #ff0000; color: white; padding: 5px; border-radius: 3px;">Dark Red</span>: 
                    Nucleotides that <strong>promote Fragile Site</strong> classification
                </p>
                <p>
                    <span style="background-color: #ccccff; padding: 5px; border-radius: 3px;">Light Blue</span> → 
                    <span style="background-color: #6666ff; padding: 5px; border-radius: 3px;">Medium Blue</span> → 
                    <span style="background-color: #0000ff; color: white; padding: 5px; border-radius: 3px;">Dark Blue</span>: 
                    Nucleotides that <strong>promote Non-Fragile</strong> classification
                </p>
                <p><strong>💡 Tip:</strong> Hover over any nucleotide to see its position, base, and exact SHAP value.</p>
                <p><strong>Average Absolute SHAP Value:</strong> {avg_importance:.4f}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(html_content)
# SHAP ANALYSIS FUNCTIONS

def create_shap_explainer(model, tokenizer, background_sequences, explainer_type='partition', max_length=6144):
    """Create SHAP explainer for CNN model"""
    print(f"Creating {explainer_type} SHAP explainer...")
    
    model_wrapper = CNNModelWrapper(model, tokenizer, max_length)
    
    if explainer_type == 'partition':
        # Tokenize background sequences to create proper input format
        encoded = tokenizer(background_sequences[:50], max_length=max_length, 
                          padding='max_length', truncation=True)
        input_ids = np.array(encoded['input_ids'])
        
        # Create partition masker
        masker = shap.maskers.Partition(input_ids)
        
        # Initialize SHAP explainer with partition algorithm
        explainer = shap.Explainer(
            model_wrapper,
            masker=masker,
            algorithm='partition',
            max_evals=250,
            batch_size=1,
            silent=True
        )
        
        # Set background data
        explainer.masker.data = input_ids
        explainer.masker._shape = input_ids.shape
        
        return explainer
    
    elif explainer_type == 'permutation':
        # Create permutation explainer
        explainer = shap.PermutationExplainer(
            model_wrapper,
            background_sequences[:100],
            max_evals=250,
            seed=42
        )
        
        return explainer
    
    elif explainer_type == 'kernel':
        # Create kernel explainer
        explainer = shap.KernelExplainer(
            model_wrapper,
            background_sequences[:50]
        )
        
        return explainer
    
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

def analyze_sequence_with_shap(explainer, sequences, tokenizer, max_length=6144, sequence_names=None):
    """Analyze sequences using SHAP"""
    print("Analyzing sequences with SHAP...")
    
    if isinstance(sequences, str):
        sequences = [sequences]
    
    if sequence_names is None:
        sequence_names = [f"sequence_{i}" for i in range(len(sequences))]
    
    results = []
    
    for i, (sequence, name) in enumerate(zip(sequences, sequence_names)):
        print(f"Analyzing sequence {i+1}/{len(sequences)}: {name}")
        
        try:
            # Get SHAP values 
            encoded = tokenizer(sequence, max_length=max_length, padding='max_length', truncation=True)
            input_ids = np.array([encoded['input_ids']])  
            shap_values = explainer(input_ids)

            
            if hasattr(shap_values, 'values'):
                importance_scores = shap_values.values[0]
            else:
                importance_scores = shap_values[0]
            
            # Truncate sequence and scores to actual length
            actual_length = min(len(sequence), len(importance_scores))
            sequence_truncated = sequence[:actual_length]
            scores_truncated = importance_scores[:actual_length]

            # Identify top contributing positions
            top_k = 10
            importance_array = np.array(scores_truncated)
            top_indices = importance_array.argsort()[-top_k:]
            top_scores = importance_array[top_indices]

            
            results.append({
                'name': name,
                'sequence': sequence_truncated,
                'sequence_name': name,
                'fragile_importance': scores_truncated.tolist(),
                'prediction': float(explainer.model([sequence])[0]),
                'length': actual_length,
                'top_important_positions': top_indices.tolist(),
                'top_importance_scores': top_scores.tolist()
            })
            
        except Exception as e:
            print(f"Error analyzing sequence {name}: {e}")
            continue
    
    return results

def find_fragile_site_motifs(shap_results, explainer, motif_length=None):
    motif_importance = {}
    if not shap_results:
        print("No SHAP results available for motif analysis")
        return []

    if hasattr(explainer, "analyze_length"):
        analyze_length = explainer.analyze_length
    else:
        analyze_length = min(len(res['sequence']) for res in shap_results)

    if motif_length is None or motif_length > analyze_length:
        motif_length = min(100, analyze_length)
        print(f"[INFO] Motif length automatically set to {motif_length} bp (matches SHAP explainer).")

    for result in shap_results:
        seq = result['sequence']
        importance = result['fragile_importance']

        # Convert to numpy array and ensure it's 1D
        if isinstance(importance, list):
            importance = np.array(importance)
        if importance.ndim > 1:
            importance = importance.flatten()

        if len(seq) < motif_length:
            continue
        if len(importance) == 0 or np.all(np.isnan(importance)):
            continue

        # Calculate threshold safely
        valid_importance = importance[~np.isnan(importance)]
        if len(valid_importance) == 0:
            continue
        threshold = np.percentile(valid_importance, 95)

        # Iterate through sequence positions
        for i in range(len(seq) - motif_length + 1):
            if i + motif_length <= len(importance):
                try:
                    window_importance = importance[i:i+motif_length]
                    if len(window_importance) > 0 and np.mean(window_importance) > threshold:
                        motif = seq[i:i+motif_length]
                        avg_importance = float(np.mean(window_importance))

                        if motif not in motif_importance:
                            motif_importance[motif] = []
                        motif_importance[motif].append(avg_importance)
                except Exception as e:
                    print(f"Warning: Error processing position {i}: {e}")
                    continue

    # Calculate final motif scores
    motif_scores = {}
    for motif, scores in motif_importance.items():
        if len(scores) >= 2:
            motif_scores[motif] = float(np.mean(scores))
    
    sorted_motifs = sorted(motif_scores.items(), key=lambda x: x[1], reverse=True)

    if sorted_motifs:
        print(f"\nFound {len(sorted_motifs)} recurring motifs")
        for motif, score in sorted_motifs[:10]:
            display_motif = motif[:20] + "..." + motif[-20:] if len(motif) > 40 else motif
            print(f"  {display_motif}: {score:.4f}")
    else:
        print("No recurring motifs found")

    return sorted_motifs

# VISUALIZATION AND REPORTING FUNCTIONS  

def visualize_shap_results(shap_results, save_path="shap_analysis.png"):
    """Create SHAP visualizations - fixed for array/list compatibility"""
    print("Creating SHAP visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('SHAP Analysis Results for CNN Baseline', fontsize=16)
    
    if len(shap_results) > 0:
        result = shap_results[0]
        
        # Ensure numpy array
        importance = np.array(result['fragile_importance'])
        
        # Nucleotide importance heatmap
        ax1 = axes[0, 0]
        importance_matrix = importance.reshape(1, -1)
        seq_chars = list(result['sequence'][:len(importance)])
        
        im = ax1.imshow(importance_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_title('Nucleotide Importance for Fragile Site Detection')
        ax1.set_xlabel('Nucleotide Position')
        ax1.set_ylabel('Sequence')
        
        if len(seq_chars) < 100:
            ax1.set_xticks(range(len(seq_chars)))
            ax1.set_xticklabels(seq_chars, rotation=90, fontsize=8)
        
        plt.colorbar(im, ax=ax1, label='SHAP Value')
        
        # Top important positions
        ax2 = axes[0, 1]
        top_k = min(10, len(importance))
        top_indices = np.argsort(np.abs(importance))[-top_k:]
        top_positions = top_indices
        top_scores = importance[top_indices]
        
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
        all_shap_values.extend(np.array(result['fragile_importance']).flatten())
    
    ax3.hist(all_shap_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Distribution of SHAP Values')
    ax3.set_xlabel('SHAP Value')
    ax3.set_ylabel('Frequency')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Nucleotide composition analysis
    ax4 = axes[1, 1]
    nucleotide_importance = {'A': [], 'T': [], 'G': [], 'C': []}
    
    for result in shap_results:
        seq = result['sequence']
        importance = np.array(result['fragile_importance'])
        
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

# Add this function right after your other SHAP functions (before main())
def create_shap_text_plot(shap_values, tokenizer, sequence, save_path=None):
    """
    Create SHAP text plot that works with DNA sequences
    Properly handles the data format for shap.plots.text()
    """
    print("Generating SHAP text plot for DNA sequence...")
    
    try:
        # Get actual sequence length (without padding)
        actual_length = min(len(sequence), 100)  # Limit to 100 for better visualization
        
        # Extract the token IDs and values
        if hasattr(shap_values, 'data'):
            if len(shap_values.data.shape) > 1:
                token_ids = shap_values.data[0]
            else:
                token_ids = shap_values.data
        else:
            token_ids = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        if hasattr(shap_values, 'values'):
            values = shap_values.values[0] if len(shap_values.values.shape) > 1 else shap_values.values
        else:
            values = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Convert token IDs to nucleotide strings
        tokens = []
        clean_values = []
        
        for i in range(min(len(token_ids), actual_length)):
            if isinstance(token_ids[i], (int, np.integer)):
                token = tokenizer.id_to_char.get(token_ids[i], 'N')
            else:
                token = str(token_ids[i])
            
            # Keep all nucleotides including N
            if token in ['A', 'T', 'G', 'C', 'N']:
                tokens.append(token)
                clean_values.append(values[i] if i < len(values) else 0.0)
        
        # Create clean shap values with proper format
        clean_shap_values = shap.Explanation(
            values=np.array([clean_values]),
            base_values=shap_values.base_values if hasattr(shap_values, 'base_values') else np.array([0]),
            data=np.array([tokens]),  # This should be array of strings
            feature_names=tokens
        )
        
        # Set the .data attribute to be strings
        clean_shap_values.data = np.array([tokens])
        
        # Create the text plot
        shap.plots.text(clean_shap_values)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP text plot saved to {save_path}")
        
        return True
        
    except Exception as e:
        print(f"Text plot failed with error: {e}")
        print("Creating custom DNA visualization instead...")
        
        # Custom visualization as fallback
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(20, 3))
        
        # Get values
        if hasattr(shap_values, 'values'):
            values = shap_values.values[0][:actual_length]
        else:
            values = shap_values[0][:actual_length]
        
        # Normalize values for color mapping
        max_abs_val = np.max(np.abs(values)) if len(values) > 0 else 1
        
        # Create text with colors
        for i, (nuc, val) in enumerate(zip(sequence[:actual_length], values)):
            # Determine color based on SHAP value
            if val > 0:
                color = plt.cm.Reds(abs(val) / max_abs_val)
            else:
                color = plt.cm.Blues(abs(val) / max_abs_val)
            
            ax.text(i, 0, nuc, fontsize=14, fontweight='bold',
                   color=color, ha='center', va='center',
                   family='monospace')
        
        ax.set_xlim(-0.5, actual_length - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        ax.set_title('SHAP Values for DNA Sequence', fontsize=16, pad=20)
        
        # Add colorbar legend
        sm_red = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                       norm=plt.Normalize(vmin=0, vmax=max_abs_val))
        sm_blue = plt.cm.ScalarMappable(cmap=plt.cm.Blues_r, 
                                        norm=plt.Normalize(vmin=-max_abs_val, vmax=0))
        
        # Add legend
        red_patch = mpatches.Patch(color='red', label='Increases fragile site probability')
        blue_patch = mpatches.Patch(color='blue', label='Decreases fragile site probability')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Custom DNA visualization saved to {save_path}")
        
        plt.show()
        
        return False
    
def generate_shap_report(shap_results, model_performance, save_path="shap_report.txt"):
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
        # Convert to numpy array if it's a list
        importance = np.array(result['fragile_importance'])
        all_importance.extend(importance)
        
        for i, nuc in enumerate(seq):
            if i < len(importance) and nuc in nucleotide_counts:
                nucleotide_counts[nuc] += 1
                nucleotide_importance[nuc].append(importance[i])
    
    for result in shap_results:
        seq_length = len(result['sequence'])
        # Convert to numpy array for operations
        importance_array = np.array(result['fragile_importance'])
        positive_shap = np.sum(importance_array[importance_array > 0])
        normalized_score = (positive_shap / seq_length) * 1000 if seq_length > 0 else 0
        result['normalized_score'] = normalized_score
        result['total_positive_shap'] = positive_shap
    
    report.append("TOP SEQUENCES WITH HIGHEST FRAGILE SITE SIGNALS (NORMALIZED PER KB):")
    
    seq_scores = []
    for result in shap_results:
        seq_scores.append((
            result['sequence_name'], 
            result['total_positive_shap'],
            result['normalized_score'],
            result
        ))
    
    seq_scores.sort(key=lambda x: x[2], reverse=True)
    
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
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    if verbose:
        print(f"\nFinal split statistics:")
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

def plot_roc_auc_curve(y_true, y_scores, save_path="roc_auc_curve.png", model_name="CNN"):
    """
    Plot ROC curve with AUC score for model predictions.
    """
    print(f"\nGenerating 'How well does our model work?' plot for {model_name}...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot with enhanced styling
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve with gradient effect
    plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
             label=f'Our Model (AUC = {roc_auc:.3f})', alpha=0.8) 
    
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

def run_visualization_pipeline(model, train_loader, val_loader, test_loader, 
                              device, output_dir, max_samples=2000):
    
    print("\n" + "="*70)
    print("STARTING EMBEDDING VISUALIZATION PIPELINE")
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
            
            ax.set_title(f'{dataset_name.capitalize()} Dataset', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{method}1')
            ax.set_ylabel(f'{method}2')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{method} Embeddings Across Datasets', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        combined_path = os.path.join(save_dir, f'{method.lower()}_all_datasets.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} combined dataset plot to {combined_path}")
        plt.show()

# MAIN FUNCTION

def main():
    parser = argparse.ArgumentParser(description='CNN Baseline SHAP Analysis for DNA Fragile Sites Detection')
    parser.add_argument('--bidirectional', action='store_true',
                   help='Use bidirectional CNN architecture')
    parser.add_argument('--bidirectional_strategy', type=str, 
                   default='concat', choices=['concat', 'add', 'max'],
                   help='Strategy for combining bidirectional features')
    # Data parameters 
    parser.add_argument('--pos_fasta', type=str, required=True,
                       help='Path to positive sequences FASTA file')
    parser.add_argument('--neg_fasta', type=str, required=True,
                       help='Path to negative sequences FASTA file')
    parser.add_argument('--balance_strategy', type=str, default='combined',
                       choices=['stratified', 'undersample', 'oversample', 'combined'],
                       help='Class balancing strategy (default: combined)')
    
    # Model architecture parameters  
    parser.add_argument('--max_length', type=int, default=6144,
                       help='Maximum sequence length (default: 6144)')
    parser.add_argument('--embedding_dim', type=int, default=100,
                       help='Embedding dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=12,
                       help='Vocabulary size (default: 12)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--num_filters', type=int, default=128,
                        help='Number of filters per convolution layer (default: 128)')
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[3, 7, 15, 31, 63],
                        help='List of filter sizes for convolution layers (default: [3, 5, 7])')


    # Training parameters 
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='Weight decay (default: 1e-3)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                       help='Gradient clipping max norm (default: 0.5)')
    
    # SHAP parameters
    parser.add_argument('--enable_shap', action='store_true',
                       help='Enable SHAP analysis (default: False)')
    parser.add_argument('--shap_explainer', type=str, default='partition',
                       choices=['permutation', 'partition', 'kernel'],
                       help='SHAP explainer type (default: partition)')
    parser.add_argument('--shap_background_size', type=int, default=75,
                       help='Number of background sequences for SHAP (default: 75)')
    parser.add_argument('--shap_analysis_size', type=int, default=150,
                       help='Number of sequences to analyze with SHAP (default: 150)')
    parser.add_argument('--shap_tp_samples', type=int, default=25,
                       help='Number of true positive samples for SHAP (default: 25)')
    parser.add_argument('--shap_tn_samples', type=int, default=15,
                       help='Number of true negative samples for SHAP (default: 15)')
    parser.add_argument('--shap_fp_samples', type=int, default=15,
                       help='Number of false positive samples for SHAP (default: 15)')
    parser.add_argument('--shap_fn_samples', type=int, default=15,
                       help='Number of false negative samples for SHAP (default: 15)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./CNN_Results',
                       help='Output directory (default: ./CNN_Results)')
    parser.add_argument('--model_save_name', type=str, default='cnn_fragile_sites_model.pth',
                       help='Model save filename (default: cnn_fragile_sites_model.pth)')
    parser.add_argument('--shap_plot_name', type=str, default='cnn_shap_analysis.png',
                       help='SHAP visualization save filename (default: cnn_shap_analysis.png)')
    parser.add_argument('--shap_report_name', type=str, default='cnn_shap_report.txt',
                       help='SHAP report save filename (default: cnn_shap_report.txt)')
    parser.add_argument('--shap_data_name', type=str, default='cnn_shap_data.pth',
                       help='SHAP data save filename (default: cnn_shap_data.pth)')
    parser.add_argument('--enable_visualization', action='store_true',
                        help='Enable UMAP/t-SNE/PCA visualization of embeddings')
    parser.add_argument('--viz_max_samples', type=int, default=2000,
                        help='Maximum samples for visualization (default: 2000)')
    
    args = parser.parse_args()
    
    # Create output directory
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data from FASTA files
    print("Loading data from FASTA files...")
    sequences, labels, names = load_and_create_dataset(args.pos_fasta, args.neg_fasta)
    
    # Balance dataset and create splits
    (X_train, y_train, names_train), (X_val, y_val, names_val), (X_test, y_test, names_test) = balance_and_split_data(
        sequences, labels, names, 
        balance_strategy=args.balance_strategy, 
        random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = CNNDNATokenizer()
    
    # Create datasets and dataloaders
    train_df = pd.DataFrame({'Sequence': X_train, 'Label': y_train})
    val_df = pd.DataFrame({'Sequence': X_val, 'Label': y_val})
    test_df = pd.DataFrame({'Sequence': X_test, 'Label': y_test})
    
    train_dataset, train_loader = create_simple_dataset(train_df, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_dataset, val_loader = create_simple_dataset(val_df, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_dataset, test_loader = create_simple_dataset(test_df, tokenizer, args.max_length, args.batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize resource monitoring
    resource_stats = {
        'epoch_times': [],
        'gpu_memory_used': [],
        'gpu_memory_total': [],
        'cpu_percent': [],
        'ram_percent': [],
        'ram_used_gb': []
    }

    # Initialize GPU monitoring
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("GPU monitoring initialized")
        except:
            print("GPU monitoring not available")
            gpu_handle = None
    else:
        gpu_handle = None

    start_training_time = time.time()
    print("Starting resource monitoring...")

    # Initialize CNN model
    if args.bidirectional:
        class BidirectionalCNNWithEmbedding(nn.Module):
            def __init__(self, num_filters=128, filter_sizes=[3, 7, 15, 31, 63], 
                        num_classes=2, dropout=0.5, vocab_size=12, embedding_dim=100,
                        bidirectional_strategy='concat'):
                super(BidirectionalCNNWithEmbedding, self).__init__()
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.bidirectional_strategy = bidirectional_strategy
                
                # Forward and reverse convolutions
                self.forward_convs = nn.ModuleList([
                    nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, 
                            padding=fs//2) for fs in filter_sizes
                ])
                
                self.reverse_convs = nn.ModuleList([
                    nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, 
                            padding=fs//2) for fs in filter_sizes
                ])
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                
                # Calculate embedding size based on strategy
                if bidirectional_strategy == 'concat':
                    self.embedding_size = num_filters * len(filter_sizes) * 2
                else:
                    self.embedding_size = num_filters * len(filter_sizes)
                    
                self.fc = nn.Linear(self.embedding_size, num_classes)
                
            def extract_embeddings(self, x):
                """Extract bidirectional embeddings"""
                x = self.embedding(x)
                x = x.transpose(1, 2)
                
                # Forward processing
                forward_features = []
                for conv in self.forward_convs:
                    h = self.relu(conv(x))
                    h = F.max_pool1d(h, h.size(2)).squeeze(2)
                    forward_features.append(h)
                
                # Reverse processing
                x_reversed = torch.flip(x, dims=[2])
                reverse_features = []
                for conv in self.reverse_convs:
                    h = self.relu(conv(x_reversed))
                    h = F.max_pool1d(h, h.size(2)).squeeze(2)
                    reverse_features.append(h)
                
                # Combine based on strategy
                if self.bidirectional_strategy == 'concat':
                    combined = torch.cat(forward_features + reverse_features, dim=1)
                elif self.bidirectional_strategy == 'add':
                    forward_combined = torch.cat(forward_features, dim=1)
                    reverse_combined = torch.cat(reverse_features, dim=1)
                    combined = forward_combined + reverse_combined
                elif self.bidirectional_strategy == 'max':
                    forward_combined = torch.cat(forward_features, dim=1)
                    reverse_combined = torch.cat(reverse_features, dim=1)
                    combined = torch.max(forward_combined, reverse_combined)
                
                return combined
            
            def forward(self, x):
                embeddings = self.extract_embeddings(x)
                out = self.dropout(embeddings)
                return self.fc(out)
                
    if args.bidirectional:
        model = BidirectionalCNNWithEmbedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            num_classes=2,
            dropout=args.dropout,
            bidirectional_strategy=args.bidirectional_strategy
        ).to(device)
    else:
        model = CNN(
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            num_classes=2,
            dropout=args.dropout,
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim
        ).to(device)

    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with early stopping 
    # Training loop with progress bars
    print("Training CNN model...")
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Create overall epoch progress bar
    epoch_pbar = tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        epoch_start_time = time.time()

        # Record resources at epoch start
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        ram_percent = memory_info.percent
        ram_used_gb = memory_info.used / (1024**3)
        
        # GPU monitoring
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)

        # Training phase
        model.train()
        total_train_loss = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training", 
                        leave=False, unit="batch")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            total_train_loss += loss.item()
            
            # Update training progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation", 
                    leave=False, unit="batch")
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                
                # Update validation progress bar
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Store resource stats
        resource_stats['epoch_times'].append(epoch_duration)
        resource_stats['gpu_memory_used'].append(gpu_memory_used)
        resource_stats['gpu_memory_total'].append(gpu_memory_total)
        resource_stats['cpu_percent'].append(cpu_percent)
        resource_stats['ram_percent'].append(ram_percent)
        resource_stats['ram_used_gb'].append(ram_used_gb)
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Loss': f'{avg_val_loss:.4f}',
            'Val Acc': f'{val_accuracy:.4f}',
            'GPU': f'{gpu_memory_used:.1f}GB',
            'Time': f'{epoch_duration:.1f}s'
        })
        
        # Print detailed epoch summary
        tqdm.write(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        tqdm.write(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        tqdm.write(f"  Time: {epoch_duration:.2f}s | GPU: {gpu_memory_used:.2f}/{gpu_memory_total:.2f}GB | RAM: {ram_percent:.1f}%")
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(args.output_dir, 'best_' + args.model_save_name)
            torch.save(model.state_dict(), best_model_path)
            tqdm.write(f"  ✓ New best model saved (Val Acc: {val_accuracy:.4f})")
        else:
            patience_counter += 1
            tqdm.write(f"  Patience: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            tqdm.write(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break

    epoch_pbar.close()
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    final_test_accuracy = accuracy_score(all_labels, all_predictions)
    all_probabilities = np.array(all_probabilities)
    positive_class_probs = all_probabilities[:, 1] 
    auc_score = roc_auc_score(all_labels, positive_class_probs) 
    
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    # After model evaluation (around line where you calculate final_test_accuracy)
    if args.enable_visualization:
        print("\n" + "="*70)
        print("STARTING EMBEDDING VISUALIZATION")
        print("="*70)
        
        # NO IMPORT NEEDED - functions are already in this file!
        visualization_results = run_visualization_pipeline(
            model,  # Your trained model
            train_loader,
            val_loader,
            test_loader,
            device,
            args.output_dir,
            max_samples=args.viz_max_samples
        )

        roc_save_path = os.path.join(args.output_dir, 'roc_auc_curve.png')
        auc_score, optimal_threshold = plot_roc_auc_curve(
            all_labels, 
            positive_class_probs,
            save_path=roc_save_path,
            model_name="CNN"
        )
        
        print("\nVisualization complete!")
        print(f"Check {args.output_dir}/embedding_visualizations/ for results")
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Non-Fragile', 'Fragile']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    
    # Remove the duplicate visualization code that was causing issues
    total_training_time = time.time() - start_training_time
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fragile', 'Fragile'],
                yticklabels=['Non-Fragile', 'Fragile'])
    plt.title('CNN Baseline - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    total_training_time = time.time() - start_training_time

    # Create resource usage report
    resource_report_path = os.path.join(args.output_dir, 'resource_usage_report.txt')
    with open(resource_report_path, 'w') as f:
        f.write("TRAINING RESOURCE USAGE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        f.write(f"Average Epoch Time: {np.mean(resource_stats['epoch_times']):.2f} seconds\n")
        f.write(f"Max GPU Memory Used: {max(resource_stats['gpu_memory_used']):.2f} GB\n")
        f.write(f"Average GPU Memory Used: {np.mean(resource_stats['gpu_memory_used']):.2f} GB\n")
        f.write(f"Peak CPU Usage: {max(resource_stats['cpu_percent']):.1f}%\n")
        f.write(f"Average CPU Usage: {np.mean(resource_stats['cpu_percent']):.1f}%\n")
        f.write(f"Peak RAM Usage: {max(resource_stats['ram_percent']):.1f}% ({max(resource_stats['ram_used_gb']):.2f} GB)\n")
        f.write(f"Average RAM Usage: {np.mean(resource_stats['ram_percent']):.1f}% ({np.mean(resource_stats['ram_used_gb']):.2f} GB)\n")

    print(f"Resource usage report saved to {resource_report_path}")
    
    # Save final model
    model_path = os.path.join(args.output_dir, args.model_save_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    

    if args.enable_shap:
        print("\n" + "="*70)
        print("STARTING SHAP ANALYSIS WITH HTML VISUALIZATION")
        print("="*70)
        
        # Select ONE sequence for HTML visualization (just like Caduceus script)
        sequence_for_html = X_test[0]  # Use first test sequence
        
        # Create THE SINGLE HTML visualization file
        success = create_shap_html_visualization(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence_for_html,  # Note: single sequence, not a list
            output_dir=args.output_dir,
            max_length=256
        )
        
        if success:
            print("\n✓ SHAP HTML visualization created successfully!")
            print(f"Open {args.output_dir}/shap_dna_visualization.html in your browser")
        
        # Continue with standard SHAP analysis for multiple sequences
        try:
            print("\nPerforming additional SHAP analysis...")
            
            # Create standard SHAP explainer for batch analysis
            background_sequences = X_test[:args.shap_background_size]
            explainer = create_shap_explainer(
                model, tokenizer, background_sequences,
                explainer_type=args.shap_explainer, max_length=256
            )
            
            # Analyze multiple sequences for report
            shap_results = analyze_sequence_with_shap(
                explainer, X_test[:10], tokenizer,
                max_length=256, sequence_names=[f"test_seq_{i}" for i in range(10)]
            )
            
            # Create standard 4-panel visualization
            shap_plot_path = os.path.join(args.output_dir, args.shap_plot_name)
            visualize_shap_results(shap_results, save_path=shap_plot_path)
            print(f"Standard SHAP visualization saved to {shap_plot_path}")
            
            # Generate report
            model_performance = {
                'test_accuracy': final_test_accuracy,
                'roc_auc': auc_score,
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss': best_val_loss,
                'optimal_threshold': optimal_threshold 
            }
            
            report_path = os.path.join(args.output_dir, args.shap_report_name)
            generate_shap_report(shap_results, model_performance, save_path=report_path)
            print(f"SHAP report saved to {report_path}")
            
        except Exception as e:
            print(f"Additional SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("\nSHAP analysis disabled.")

    print(f"\nTraining and analysis complete!")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Final ROC-AUC: {auc_score:.4f}")

if __name__ == "__main__":
    main()