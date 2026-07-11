import argparse
import time  
import psutil  
from Bio import SeqIO
import pandas as pd
import torch
import os
import numpy as np
import datasets
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.ndimage import gaussian_filter1d
from utils import create_df, create_df_full, remove_duplicates, create_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from shap.maskers import Masker
from types import SimpleNamespace
import shap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Caduceus Training with SHAP Analysis for Fragile Site Detection')
    
    # Data parameters
    parser.add_argument('--pos_fasta', type=str, 
                       default="/mnt/dc/ta/GLM_HFS/data/merged_fragiles.fa",
                       help='Path to positive sequences FASTA file')
    parser.add_argument('--neg_fasta', type=str,
                       default="/mnt/dc/ta/GLM_HFS/code/reference/negative_training_sequences.fa",
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
                       help='Dataset balancing strategy (default: undersample)')
    
    # Model parameters
    parser.add_argument('--model_name', type=str,
                       default="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
                       help='Caduceus model name from HuggingFace')
    parser.add_argument('--classifier_dropout', type=float, default=0.3,
                       help='Dropout rate for classifier head (default: 0.3)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization (default: 512)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Maximum number of epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--lr_patience', type=int, default=3,
                       help='Learning rate scheduler patience (default: 3)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='Learning rate reduction factor (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0)')
    
    # SHAP parameters
    parser.add_argument('--enable_shap', action='store_true',
                       help='Enable SHAP analysis (default: False)')
    parser.add_argument('--shap_explainer', type=str, default='partition',choices=['permutation', 'partition', 'kernel'],
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
    parser.add_argument('--model_save_name', type=str, default='fragile_sites_caduceus_model.pth',
                       help='Model save filename (default: fragile_sites_caduceus_model.pth)')
    parser.add_argument('--history_save_name', type=str, default='training_history.pth',
                       help='Training history save filename (default: training_history.pth)')
    parser.add_argument('--shap_plot_name', type=str, default='fragile_sites_shap_analysis.png',
                       help='SHAP visualization save filename (default: fragile_sites_shap_analysis.png)')
    parser.add_argument('--shap_report_name', type=str, default='fragile_sites_shap_report.txt',
                       help='SHAP report save filename (default: fragile_sites_shap_report.txt)')
    parser.add_argument('--shap_data_name', type=str, default='fragile_sites_shap_data.pth',
                       help='SHAP data save filename (default: fragile_sites_shap_data.pth)')
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

class CaduceusModelWithEmbeddings(torch.nn.Module):
    """Wrapper to extract embeddings from Caduceus model"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def extract_embeddings(self, input_ids, attention_mask=None):
        """Extract embeddings before the classification head"""
        # Get the transformer outputs
        outputs = self.base_model.base_model(input_ids=input_ids)
        
        # Get the last hidden states
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        
        # Pool the hidden states (mean pooling over sequence length)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = hidden_states.mean(dim=1)
        
        return embeddings
    
    def forward(self, input_ids, attention_mask=None):
        """Normal forward pass for classification"""
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)

# Embedding visualization functions
def extract_embeddings_from_dataloader(model, dataloader, device, max_samples=None):
    """Extract embeddings from a dataloader using Caduceus model"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_predictions = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            labels = batch['labels'].to(device)
            
            # Get embeddings
            embeddings = model.extract_embeddings(input_ids, attention_mask)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            _, predicted = torch.max(logits, 1)
            
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
    """Apply dimensionality reduction to embeddings"""
    
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
    """Create visualization plots for embeddings (Caduceus titles)"""
    
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
            
            # Highlight misclassifications 
            misclassified = labels != predictions
            if np.any(misclassified):
                ax2.scatter(reduced[misclassified, 0], reduced[misclassified, 1],
                           marker='x', c='red', s=50, label='Misclassified', 
                           alpha=0.8, linewidth=0.5)
            
            ax2.set_title(f'{method} - Predicted Labels', fontsize=14, fontweight='bold')
            ax2.set_xlabel(f'{method}1', fontsize=12)
            ax2.set_ylabel(f'{method}2', fontsize=12)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
    
    # Updated suptitle (Caduceus wording)
    plt.suptitle('Caduceus Model Embedding Space Visualization', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()  
    
    # Save combined plot
    combined_path = os.path.join(save_dir, 'all_methods_visualization.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to {combined_path}")
    plt.show()
    
    # Create individual plots for each method
    for method in methods:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        reduced = reduced_embeddings[method]
        
        # True labels
        ax1 = axes[0]
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
        
        # Predicted labels
        if predictions is not None:
            ax2 = axes[1]
            for label in np.unique(predictions):
                mask = predictions == label
                ax2.scatter(reduced[mask, 0], reduced[mask, 1], 
                           c=[colors_pred[label]], 
                           label=f'Pred: {"Fragile" if label == 1 else "Non-Fragile"}',
                           alpha=0.7, s=50, edgecolors='white', linewidth=1.5)
            
            # Misclassifications 
            misclassified = labels != predictions
            if np.any(misclassified):
                ax2.scatter(reduced[misclassified, 0], reduced[misclassified, 1],
                           marker='x', c='red', s=50, label='Misclassified', 
                           alpha=0.8, linewidth=0.5)
            
            ax2.set_title(f'{method} Embedding Space - Predicted Labels', fontsize=16, fontweight='bold')
            ax2.set_xlabel(f'{method} Component 1', fontsize=14)
            ax2.set_ylabel(f'{method} Component 2', fontsize=14)
            ax2.legend(loc='best', fontsize=12, frameon=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Updated suptitle for Caduceus wording
        plt.suptitle(f'{method} Visualization - Caduceus DNA Fragile Sites Model', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()  
        
        # Save individual plot
        individual_path = os.path.join(save_dir, f'{method.lower()}_visualization.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} visualization to {individual_path}")
        plt.show()
    
    return reduced_embeddings

def run_visualization_pipeline(model, train_loader, val_loader, test_loader, 
                             device, output_dir, max_samples=2000):
    """Run the complete embedding visualization pipeline"""
    
    print("\n" + "="*70)
    print("STARTING EMBEDDING VISUALIZATION PIPELINE - CADUCEUS MODEL")
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
    """Create a combined plot showing all datasets side by side"""
    
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
        
        plt.suptitle(f'Caduceus Model - {method} Embeddings Across Datasets', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        combined_path = os.path.join(save_dir, f'{method.lower()}_all_datasets.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"Saved {method} combined dataset plot to {combined_path}")
        plt.show()

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

class CaduceusModelWrapper:
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def __call__(self, inputs):
        # Handle different input types
        if isinstance(inputs, str):
            sequences = [inputs]
        elif isinstance(inputs, list):
            sequences = inputs
        elif hasattr(inputs, '__iter__'):
            sequences = list(inputs)
        else:
            sequences = [inputs]
        
        # Ensure all sequences are strings
        sequences = [str(seq) for seq in sequences]
        
        try:
            # Tokenize DNA sequences
            encoded = self.tokenizer(
                sequences,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Return only positive class probabilities for SHAP
            return probabilities[:, 1].cpu().numpy()
            
        except Exception as e:
            print(f"Error in model wrapper: {e}")
            return np.array([0.5] * len(sequences))

def create_shap_explainer(model, tokenizer, background_sequences, explainer_type='permutation', max_length=512):
    """Create a working SHAP explainer for DNA sequences"""
    print(f"Creating {explainer_type} SHAP explainer...")
    
    class SimpleDNAExplainer:
        def __init__(self, model, tokenizer, max_length):
            self.model = model
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = next(model.parameters()).device
            self.model.eval()
            self.window_size = 1  # Analyze windows instead of individual nucleotides
        
        def predict(self, sequences):
            if isinstance(sequences, str):
                sequences = [sequences]
            
            encoded = self.tokenizer(
                sequences,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(input_ids=encoded['input_ids'].to(self.device))
                probs = torch.softmax(outputs.logits, dim=1)
                return probs[:, 1].cpu().numpy()
        
        def __call__(self, sequences):
            if isinstance(sequences, str):
                sequences = [sequences]
            
            print(f"Analyzing {len(sequences)} sequences...")
            results = []
            
            for seq_idx, seq in enumerate(sequences):
                # Limit analysis to first 200 nucleotides for speed
                analyze_length = min(len(seq), 200)
                seq_to_analyze = seq[:analyze_length]
                
                print(f"  Sequence {seq_idx+1}/{len(sequences)} (analyzing first {analyze_length} nucleotides)")
                
                # Get baseline
                baseline = self.predict([seq_to_analyze])[0]
                
                # Calculate importance using windows
                importance = []
                for i in range(0, analyze_length, self.window_size):
                    masked = list(seq_to_analyze)
                    # Mask the window
                    for j in range(i, min(i + self.window_size, analyze_length)):
                        masked[j] = 'N'
                    masked_seq = ''.join(masked)
                    
                    masked_pred = self.predict([masked_seq])[0]
                    window_imp = (baseline - masked_pred) / self.window_size
                    
                    # Assign importance to each position in window
                    for j in range(i, min(i + self.window_size, analyze_length)):
                        importance.append(window_imp)
                
                # Pad with zeros for the rest of the sequence
                importance.extend([0] * (len(seq) - analyze_length))
                
                result = SimpleNamespace(
                    values=np.array(importance[:len(seq)]),
                    base_values=baseline,
                    data=seq
                )
                results.append(result)
            
            print("Analysis complete!")
            return results
    
    return SimpleDNAExplainer(model, tokenizer, max_length)

class DNAOcclusionExplainer:
    """Simple occlusion-based explainer for DNA sequences"""
    def __init__(self, model_wrapper, window_size=5):
        self.model_wrapper = model_wrapper
        self.window_size = window_size
    
    def __call__(self, sequences):
        """Calculate importance scores using occlusion"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        results = []
        for seq in sequences:
            # Get baseline prediction
            baseline_prob = self.model_wrapper([seq])[0]
            
            # Calculate importance for each position
            importance_scores = []
            seq_array = list(seq)
            
            for i in range(len(seq)):
                # Mask position with N
                masked_seq = seq_array.copy()
                start = max(0, i - self.window_size // 2)
                end = min(len(seq), i + self.window_size // 2 + 1)
                
                for j in range(start, end):
                    masked_seq[j] = 'N'
                
                masked_str = ''.join(masked_seq)
                masked_prob = self.model_wrapper([masked_str])[0]
                
                # Importance is the change in prediction
                importance = baseline_prob - masked_prob
                importance_scores.append(importance)
            
            # Create SHAP-like result object
            result = SimpleNamespace(
                values=np.array(importance_scores),
                base_values=baseline_prob,
                data=seq
            )
            results.append(result)
        
        return results

def analyze_sequence_with_shap(explainer, sequences, sequence_names=None):
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
    print("Generating SHAP interpretation report...")
    
    report = []
    report.append("="*80)
    report.append("SHAP INTERPRETATION REPORT FOR FRAGILE SITES DETECTION")
    report.append("="*80)
    report.append("")
    
    report.append("MODEL PERFORMANCE SUMMARY:")
    report.append(f"  Test Accuracy: {model_performance.get('test_accuracy', 'N/A'):.4f}")
    report.append(f"  ROC-AUC Score: {model_performance.get('roc_auc', 'N/A'):.4f}")
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

        
    # Change this section:
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
        report.append(f"     Normalized SHAP (per kb): {norm_score:.4f}")  # Add this line
        report.append(f"     Sequence length: {len(result['sequence'])}")

    report.append("NUCLEOTIDE IMPORTANCE ANALYSIS:")
    for nuc in ['A', 'T', 'G', 'C']:
        if nucleotide_importance[nuc]:
            mean_imp = np.mean(nucleotide_importance[nuc])
            std_imp = np.std(nucleotide_importance[nuc])
            report.append(f"  {nuc}: Mean SHAP = {mean_imp:.4f} (±{std_imp:.4f})")
    report.append("")
    
    report.append("TOP SEQUENCES WITH HIGHEST FRAGILE SITE SIGNALS:")
    
    seq_scores = []
    for result in shap_results:
        positive_shap = np.sum(result['fragile_importance'][result['fragile_importance'] > 0])
        # Only create 3-element tuples to match unpacking
        seq_scores.append((result['sequence_name'], positive_shap, result))
    
    # Sort by positive SHAP (index 1)
    seq_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Now unpack 3 elements correctly
    for i, (seq_name, score, result) in enumerate(seq_scores[:5]):
        report.append(f"  {i+1}. {seq_name}")
        report.append(f"     Total positive SHAP: {score:.4f}")
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
        print(f"CREATING ENHANCED BALANCED SPLITS (Strategy: {balance_strategy})")
    
    # Get initial class distribution
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
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}. "
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
    """Find sequence motifs associated with high SHAP values"""
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
    
    # Calculate average importance per motif
    motif_scores = {
        motif: np.mean(scores) 
        for motif, scores in motif_importance.items() 
        if len(scores) >= 2  # Seen at least twice
    }
    
    # Sort by importance
    sorted_motifs = sorted(motif_scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_motifs:
        print(f"\nFound {len(sorted_motifs)} recurring motifs")
        print("\nTop 10 Fragile Site Motifs:")
        for motif, score in sorted_motifs[:10]:
            print(f"  {motif[:20]}...{motif[-20:]}: {score:.4f}")
    else:
        print("No recurring motifs found")
    
    return sorted_motifs

def create_shap_html_visualization(model, tokenizer, sequence, output_dir, max_length=256):
    """
    Create SHAP HTML visualization for Caduceus model
    """
    # Import all required modules at the function level
    import shap
    import numpy as np
    import torch
    import os  # Make sure this import is here
    
    print("\n" + "="*60)
    print("CREATING SHAP HTML VISUALIZATION (Caduceus)")
    print("="*60)
    
    # Create model wrapper for HTML visualization
    class CaduceusHTMLWrapper:
        def __init__(self, model, tokenizer, max_length, device):
            self.model = model
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device
            self.model.eval()
        
        def __call__(self, sequences):
            if isinstance(sequences, str):
                sequences = [sequences]
            elif isinstance(sequences, np.ndarray):
                sequences = sequences.tolist() if sequences.ndim > 1 else [sequences]
            
            # Clean sequences
            cleaned = []
            for seq in sequences:
                if isinstance(seq, (list, np.ndarray)):
                    seq = ''.join(str(s) for s in seq)
                cleaned.append(str(seq).replace('[MASK]', 'N').replace('<mask>', 'N'))
            
            # Tokenize and predict
            encoded = self.tokenizer(
                cleaned,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            return probs.cpu().numpy()
    
    # Setup
    device = next(model.parameters()).device
    wrapper = CaduceusHTMLWrapper(model, tokenizer, max_length, device)
    
    # Truncate sequence if needed
    if len(sequence) > 500:
        sequence_to_analyze = sequence[:500]
        print(f"Sequence truncated to 500bp for visualization")
    else:
        sequence_to_analyze = sequence
    
    print(f"Analyzing sequence of length {len(sequence_to_analyze)}...")
    
    # Calculate importance scores using window-based approach
    print("Computing importance scores...")
    
    baseline_probs = wrapper([sequence_to_analyze])[0]
    baseline_score = baseline_probs[1]  # Fragile class
    
    importance_scores = []
    window_size = 5
    
    for i in range(0, len(sequence_to_analyze), window_size):
        masked_seq = list(sequence_to_analyze)
        for j in range(i, min(i + window_size, len(sequence_to_analyze))):
            masked_seq[j] = 'N'
        masked_seq = ''.join(masked_seq)
        
        masked_probs = wrapper([masked_seq])[0]
        masked_score = masked_probs[1]
        
        importance = baseline_score - masked_score
        
        for j in range(i, min(i + window_size, len(sequence_to_analyze))):
            if j < len(sequence_to_analyze):
                importance_scores.append(importance)
    
    importance_scores = importance_scores[:len(sequence_to_analyze)]
    
    import os  
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "caduceus_shap_dna_visualization.html")
    
    try:
        # Try SHAP's built-in visualization
        shap_values = shap.Explanation(
            values=np.array([importance_scores]),
            base_values=np.array([baseline_score]),
            data=np.array([list(sequence_to_analyze)]),
            feature_names=list(sequence_to_analyze)
        )
        
        html_output = shap.plots.text(shap_values, display=False)
        
        if isinstance(html_output, str):
            html_string = html_output
        elif hasattr(html_output, 'data'):
            html_string = html_output.data
        else:
            html_string = str(html_output)
        
        with open(html_path, 'w') as f:
            f.write(html_string)
        
        print(f"✓ SHAP visualization saved to: caduceus_shap_dna_visualization.html")
        return True
        
    except Exception as e:
        print(f"Creating custom HTML visualization: {e}")
        create_custom_caduceus_html(
            sequence_to_analyze,
            importance_scores,
            baseline_score,
            html_path,
            model_name="Caduceus"
        )
        print(f"Custom HTML saved to: caduceus_shap_dna_visualization.html")
        return True

def plot_roc_auc_curve(y_true, y_scores, save_path="roc_auc_curve.png", model_name="Caduceus"):
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


def create_custom_caduceus_html(sequence, importance_scores, baseline_score, output_path, model_name="Caduceus"):
    """
    Create custom HTML visualization for Caduceus (without CNN/HyenaDNA comparisons)
    """
    import numpy as np
    import os  
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    values = np.array(importance_scores)
    max_val = np.max(np.abs(values)) if len(values) > 0 and not np.all(np.isnan(values)) else 1
    if max_val == 0:
        max_val = 1
    
    # Create HTML with Caduceus-specific styling (purple gradient theme)
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
                border-left: 5px solid #9b59b6;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .legend h3 {{
                margin-top: 0;
                color: #9b59b6;
                font-size: 20px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 25px;
                margin: 30px 0;
            }}
            .stat-box {{
                padding: 20px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                transition: all 0.3s;
                text-align: center;
                border-top: 3px solid #9b59b6;
            }}
            .stat-box:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(135deg, #9b59b6, #e74c3c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{
                font-size: 13px;
                color: #7f8c8d;
                margin-top: 8px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="model-badge">🧬 {model_name} Transformer Model</div>
            <h1>DNA Sequence SHAP Analysis Dashboard</h1>
            <div class="subtitle">
                Model: {model_name} | Sequence Length: {len(sequence)}bp | 
                Baseline Probability: {baseline_score:.4f}
            </div>
            
            <div class="class-labels">
                <div class="output-0">✓ Non-Fragile Site</div>
                <div class="output-1">⚠ Fragile Site</div>
            </div>
            
            <div class="info-box">
                <strong>🔬 Analysis Details:</strong><br>
                • <strong>Model Architecture:</strong> {model_name} Transformer<br>
                • <strong>Baseline Fragile Probability:</strong> {baseline_score:.4f}<br>
                • <strong>Analysis Method:</strong> Window-based masking (5bp windows)<br>
                • <strong>Color Interpretation:</strong> Red = increases fragility | Blue = decreases fragility
            </div>
    """
    
    # Statistics
    num_promoting = np.sum(values > 0)
    num_protective = np.sum(values < 0)
    avg_importance = np.mean(np.abs(values))
    max_importance = np.max(np.abs(values)) if len(values) > 0 else 0
    
    html_content += f"""
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{len(sequence)}</div>
                    <div class="stat-label">Sequence Length</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_promoting}</div>
                    <div class="stat-label">Risk Positions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{num_protective}</div>
                    <div class="stat-label">Protective Positions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{max_importance:.3f}</div>
                    <div class="stat-label">Peak Importance</div>
                </div>
            </div>
            
            <div class="sequence-container">
    """
    
    # Add nucleotides with coloring
    for i, (nucleotide, val) in enumerate(zip(sequence, values)):
        intensity = min(abs(val) / max_val, 1.0) if max_val > 0 else 0
        
        if val > 0:  # Fragile-promoting
            r = 255
            g = int(255 * (1 - intensity * 0.9))
            b = int(255 * (1 - intensity * 0.9))
        else:  # Protective
            r = int(255 * (1 - intensity * 0.9))
            g = int(255 * (1 - intensity * 0.9))
            b = 255
        
        text_color = 'white' if intensity > 0.5 else 'black'
        
        style = f"background-color: rgb({r},{g},{b}); color: {text_color};"
        
        html_content += f'<span class="nucleotide" style="{style}" '
        html_content += f'title="Position {i+1} | {nucleotide} | Score: {val:.4f}">{nucleotide}</span>'
        
        if (i + 1) % 60 == 0:
            html_content += '<br>'
    
    html_content += f"""
            </div>
            
            <div class="legend">
                <h3>📊 Interpretation Guide - {model_name} Model</h3>
                <p><strong>Understanding the Visualization:</strong></p>
                <ul style="line-height: 1.8;">
                    <li>🔴 <strong>Red nucleotides:</strong> Increase fragile site probability (risk factors)</li>
                    <li>🔵 <strong>Blue nucleotides:</strong> Decrease fragile site probability (protective factors)</li>
                    <li>⚪ <strong>White/light colors:</strong> Minimal impact on prediction</li>
                    <li>📈 <strong>Color intensity:</strong> Stronger colors = higher importance scores</li>
                </ul>
                <p style="margin-top: 15px;">
                    <strong>Model Statistics:</strong><br>
                    • Average Absolute Importance: {avg_importance:.4f}<br>
                    • Importance Ratio (Risk/Protective): {num_promoting}/{num_protective}<br>
                    • Model Architecture: {model_name} Transformer
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    args = parse_arguments()
    
    # Set device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("STARTING CADUCEUS TRAINING WITH SHAP ANALYSIS")
    print("=" * 70)
    print(f"Arguments: {vars(args)}")
    print("=" * 70)

    # Loading the dataset with verification
    df_full = create_df_full_verified(args.pos_fasta, args.neg_fasta, args.verbose)

    if df_full is None:
        print("Failed to load data. Exiting.")
        exit(1)

    if args.verbose:
        print(f"DATASET LOADED SUCCESSFULLY")
        print(f"Total samples: {len(df_full)}")
        print(f"Positive samples: {(df_full['Label'] == 1).sum()}")
        print(f"Negative samples: {(df_full['Label'] == 0).sum()}")

    # Loading the Caduceus Model
    config_overrides = {
        "rcps": True,
        "num_labels": 2,
        "classifier_dropout": args.classifier_dropout,
    }  
    config = AutoConfig.from_pretrained(
        args.model_name,
        **config_overrides,
        trust_remote_code=True
    )

    base_model = AutoModelForSequenceClassification.from_config(
        config,
        trust_remote_code=True
    )

    model = CaduceusModelWithEmbeddings(base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )

    # CREATE BALANCED SPLITS
    train_df, val_df, test_df = enhanced_balanced_split(
        df_full, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.random_state,
        balance_strategy=args.balance_strategy,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"FINAL DATASET SIZES:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val: {len(val_df)} samples") 
        print(f"   Test: {len(test_df)} samples")

    # Calculate class weights for balanced training
    train_labels = train_df['Label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    if args.verbose:
        print(f"CLASS WEIGHTS CALCULATED:")
        print(f"   Negative (0): {class_weight_dict[0]:.3f}")
        print(f"   Positive (1): {class_weight_dict[1]:.3f}")

    # Load datasets
    train_dataset, train_dataloader = create_dataset(train_df, tokenizer)
    val_dataset, val_dataloader = create_dataset(val_df, tokenizer)
    test_dataset, test_dataloader = create_dataset(test_df, tokenizer)

    model.to(device)

    # Training parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=args.lr_factor, 
        patience=args.lr_patience, 
        min_lr=args.min_lr,
        verbose=True
    )

    # Weighted loss function
    class_weight_tensor = torch.tensor([class_weight_dict[0], class_weight_dict[1]], 
                                     dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

    if args.verbose:
        print(f"TRAINING CONFIGURATION:")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Max epochs: {args.num_epochs}")
        print(f"   Device: {device}")
        print(f"   Weighted loss: Yes (using class weights)")
        print(f"   Total training samples: {len(train_dataset)}")

    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []

    # Initialize resource monitoring
    training_resource_stats = {
        'epoch_times': [],
        'gpu_memory_used': [],
        'gpu_memory_total': [],
        'cpu_percent': [],
        'ram_percent': [],
        'ram_used_gb': []
    }

    print(f"STARTING TRAINING WITH RESOURCE MONITORING")
    print("=" * 60)

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
        if torch.cuda.is_available() and device != 'cpu':
            gpu_memory_used = torch.cuda.memory_allocated(device) / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs} [Train]')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]
            
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
            
            # Update progress bar with resource info
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_train_predictions/total_train_samples:.4f}',
                'GPU': f'{gpu_memory_used:.1f}GB'
            })

        train_loss /= len(train_dataset)
        train_accuracy = correct_train_predictions / total_train_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase (keep existing validation code)
        model.eval()
        val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{args.num_epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                correct_val_predictions += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

        val_loss /= len(val_dataset)
        val_accuracy = correct_val_predictions / total_val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Test evaluation (keep existing test code)
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]
                loss = loss_fn(logits, labels)

                test_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, 1)
                correct_test_predictions += (predicted == labels).sum().item()
                total_test_samples += labels.size(0)

        test_loss /= len(test_dataset)
        test_accuracy = correct_test_predictions / total_test_samples
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Record epoch metrics
        epoch_duration = time.time() - epoch_start_time
        training_resource_stats['epoch_times'].append(epoch_duration)
        training_resource_stats['gpu_memory_used'].append(gpu_memory_used)
        training_resource_stats['gpu_memory_total'].append(gpu_memory_total)
        training_resource_stats['cpu_percent'].append(cpu_percent)
        training_resource_stats['ram_percent'].append(ram_percent)
        training_resource_stats['ram_used_gb'].append(ram_used_gb)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Enhanced epoch summary with resource info
        print(f'Epoch {epoch + 1}/{args.num_epochs}:')
        print(f'  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        print(f'  Test  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'  Resources - Time: {epoch_duration:.2f}s | GPU: {gpu_memory_used:.2f}/{gpu_memory_total:.2f}GB | RAM: {ram_percent:.1f}%')

        # Early stopping check (keep existing code)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f'  New best validation loss: {best_val_loss:.4f}')
        else:
            epochs_without_improvement += 1
            print(f'  No improvement for {epochs_without_improvement} epochs')

        if epochs_without_improvement >= args.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            print(f'Best validation loss: {best_val_loss:.4f}')
            print(f'Best validation accuracy: {best_val_accuracy:.4f}')
            break

        print('-' * 50)

    total_training_time = time.time() - total_training_start

    # Save training resource report
    import os
    training_resource_path = os.path.join(args.output_dir, 'training_resource_report.txt')
    with open(training_resource_path, 'w') as f:
        f.write("CADUCEUS TRAINING RESOURCE USAGE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        f.write(f"Average Epoch Time: {np.mean(training_resource_stats['epoch_times']):.2f} seconds\n")
        f.write(f"Max GPU Memory Used: {max(training_resource_stats['gpu_memory_used']) if training_resource_stats['gpu_memory_used'] else 0:.2f} GB\n")
        f.write(f"Average GPU Memory Used: {np.mean(training_resource_stats['gpu_memory_used']) if training_resource_stats['gpu_memory_used'] else 0:.2f} GB\n")
        f.write(f"Peak CPU Usage: {max(training_resource_stats['cpu_percent']) if training_resource_stats['cpu_percent'] else 0:.1f}%\n")
        f.write(f"Average CPU Usage: {np.mean(training_resource_stats['cpu_percent']) if training_resource_stats['cpu_percent'] else 0:.1f}%\n")
        f.write(f"Peak RAM Usage: {max(training_resource_stats['ram_percent']) if training_resource_stats['ram_percent'] else 0:.1f}% ({max(training_resource_stats['ram_used_gb']) if training_resource_stats['ram_used_gb'] else 0:.2f} GB)\n")
        f.write(f"Average RAM Usage: {np.mean(training_resource_stats['ram_percent']) if training_resource_stats['ram_percent'] else 0:.1f}% ({np.mean(training_resource_stats['ram_used_gb']) if training_resource_stats['ram_used_gb'] else 0:.2f} GB)\n")

    print(f"Training resource report saved to: {training_resource_path}")


    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")

    # Final evaluation on test set
    model.eval()
    final_test_loss = 0.0
    correct_final_predictions = 0
    total_final_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)

            final_test_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct_final_predictions += (predicted == labels).sum().item()
            total_final_samples += labels.size(0)

    final_test_loss /= len(test_dataset)
    final_test_accuracy = correct_final_predictions / total_final_samples

    print(f'FINAL RESULTS:')
    print(f'Test Loss: {final_test_loss:.4f}')
    print(f'Test Accuracy: {final_test_accuracy:.4f}')

    # Save the best model
    import os
    model_save_path = os.path.join(args.output_dir, args.model_save_name)
    torch.save(best_model_state, model_save_path)
    print(f"Best model weights saved as '{model_save_path}'")

    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'final_test_loss': final_test_loss,
        'final_test_accuracy': final_test_accuracy,
        'args': vars(args)
    }

    history_save_path = os.path.join(args.output_dir, args.history_save_name)
    torch.save(training_history, history_save_path)
    print(f"Training history saved as '{history_save_path}'")

    # Load your best model and evaluate
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Get predictions on test set
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)

    if args.enable_visualization:
        print("\n" + "="*70)
        print("STARTING EMBEDDING VISUALIZATION")
        print("="*70)
        
        visualization_results = run_visualization_pipeline(
            model,  # The wrapped model with embedding extraction
            train_dataloader,
            val_dataloader,
            test_dataloader,
            device,
            args.output_dir,
            max_samples=args.viz_max_samples
        )
        
        print("\nVisualization complete!")
        print(f"Check {args.output_dir}/embedding_visualizations/ for results")
        
        # Save visualization data
        viz_data_path = os.path.join(args.output_dir, 'embedding_visualization_data.pth')
        torch.save(visualization_results, viz_data_path)
        print(f"Visualization data saved to {viz_data_path}")

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
            model_name="Caduceus"
        )


    # Detailed metrics
    print("=== DETAILED PERFORMANCE REPORT ===")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Non-Fragile', 'Fragile']))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # ROC-AUC Score
    auc_score = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # SHAP ANALYSIS INTEGRATION

    if args.enable_shap:
        print("\n" + "="*70)
        print("STARTING SHAP INTERPRETATION ANALYSIS")
        print("="*70)

        # Prepare data for SHAP analysis
        test_sequences = test_df['Sequence'].tolist()
        test_names = test_df['Name'].tolist()

        # Create SHAP explainer
        background_sequences = test_sequences[:args.shap_background_size]
        explainer = create_shap_explainer(
            model, 
            tokenizer, 
            background_sequences,
            args.shap_explainer,
            args.max_length
        )

        # Select representative sequences for SHAP analysis
        analysis_indices = []
        tp_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) 
                      if true == 1 and pred == 1]
        tn_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) 
                      if true == 0 and pred == 0]
        fp_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) 
                      if true == 0 and pred == 1]
        fn_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) 
                      if true == 1 and pred == 0]

        # Select representatives from each category
        analysis_indices.extend(tp_indices[:args.shap_tp_samples])
        analysis_indices.extend(tn_indices[:args.shap_tn_samples])
        analysis_indices.extend(fp_indices[:args.shap_fp_samples])
        analysis_indices.extend(fn_indices[:args.shap_fn_samples])

        # Limit to maximum specified
        analysis_indices = analysis_indices[:args.shap_analysis_size]

        # Get sequences for SHAP analysis
        shap_sequences = [test_sequences[i] for i in analysis_indices]
        shap_names = [f"{test_names[i]}_TrueLabel{all_labels[i]}_Pred{all_predictions[i]}" 
                      for i in analysis_indices]

        print(f"Analyzing {len(shap_sequences)} representative sequences with SHAP")

        # Perform SHAP analysis
        try:
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

            # Create comprehensive visualizations
            shap_plot_path = os.path.join(args.output_dir, args.shap_plot_name)
            visualization_figure = visualize_shap_results(
                shap_results, 
                save_path=shap_plot_path
            )

            # Generate detailed report
            model_performance = {
                'test_accuracy': final_test_accuracy,
                'roc_auc': auc_score,
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss': best_val_loss,
                'optimal_threshold': optimal_threshold 
            }

            shap_report_path = os.path.join(args.output_dir, args.shap_report_name)
            shap_report = generate_shap_report(
                shap_results, 
                model_performance,
                save_path=shap_report_path
            )
            print("\n" + "="*70)
            print("PERFORMING MOTIF ANALYSIS ON HIGH-IMPORTANCE REGIONS")
            print("="*70)

            # Initialize sorted_motifs to avoid undefined variable error
            sorted_motifs = []

            try:
                # Perform motif analysis
                sorted_motifs = find_fragile_site_motifs(shap_results, motif_length=600)
                
                # Only save if motifs were found
                if sorted_motifs:
                    motif_results_path = os.path.join(args.output_dir, 'fragile_site_motifs.txt')
                    with open(motif_results_path, 'w') as f:
                        f.write("FRAGILE SITE MOTIF ANALYSIS RESULTS\n")
                        f.write("="*50 + "\n\n")
                        f.write(f"Total motifs found: {len(sorted_motifs)}\n\n")
                        f.write("Top 20 Fragile Site Motifs (600bp):\n")
                        for i, (motif, score) in enumerate(sorted_motifs[:20]):
                            f.write(f"\n{i+1}. Average SHAP Score: {score:.4f}\n")
                            f.write(f"   Motif: {motif[:50]}...{motif[-50:]}\n")  # Show first and last 50bp
                            f.write(f"   Length: {len(motif)}bp\n")
                    
                    print(f"Motif analysis results saved to: {motif_results_path}")
                    print(f"Found {len(sorted_motifs)} unique motifs")
                else:
                    print("No significant motifs found in the analysis")
                    
            except Exception as e:
                print(f"Error during motif analysis: {str(e)}")
                import traceback
                traceback.print_exc()

            # Print key insights
            print("\n" + "="*70)
            print("KEY SHAP INSIGHTS")
            print("="*70)

            # Calculate overall nucleotide importance
            all_importance = []
            nucleotide_importance = {'A': [], 'T': [], 'G': [], 'C': []}

            for result in shap_results:
                seq = result['sequence']
                importance = result['fragile_importance']
                all_importance.extend(importance)
                
                for i, nuc in enumerate(seq):
                    if i < len(importance) and nuc in nucleotide_importance:
                        nucleotide_importance[nuc].append(importance[i])

            print("AVERAGE SHAP VALUES BY NUCLEOTIDE:")
            for nuc in ['A', 'T', 'G', 'C']:
                if nucleotide_importance[nuc]:
                    mean_imp = np.mean(nucleotide_importance[nuc])
                    std_imp = np.std(nucleotide_importance[nuc])
                    print(f"   {nuc}: {mean_imp:+.4f} (±{std_imp:.4f})")

            print(f"\nOVERALL STATISTICS:")
            print(f"   Total SHAP values analyzed: {len(all_importance)}")
            print(f"   Mean SHAP value: {np.mean(all_importance):.4f}")
            print(f"   Positive contributions: {np.sum(np.array(all_importance) > 0) / len(all_importance) * 100:.1f}%")
            print(f"   Negative contributions: {np.sum(np.array(all_importance) < 0) / len(all_importance) * 100:.1f}%")

            
            # Select first test sequence for HTML visualization
            html_sequence = test_df['Sequence'].iloc[0] if not test_df.empty else shap_sequences[0]
            
            # Create HTML visualization
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
                print(f"  • Caduceus: {args.output_dir}/caduceus_shap_dna_visualization.html")

                

            # Find most important sequences
            seq_scores = []
            for result in shap_results:
                positive_shap = np.sum(result['fragile_importance'][result['fragile_importance'] > 0])
                seq_scores.append((result['sequence_name'], positive_shap))

            seq_scores.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTOP 5 SEQUENCES WITH HIGHEST FRAGILE SITE SIGNALS:")
            for i, (seq_name, score) in enumerate(seq_scores[:5]):
                print(f"   {i+1}. {seq_name}: {score:.4f}")

            # Save additional SHAP data
            shap_data = {
                'shap_results': shap_results,
                'model_performance': model_performance,
                'analysis_indices': analysis_indices,
                'nucleotide_importance': nucleotide_importance,
                'args': vars(args)
            }

            shap_data_path = os.path.join(args.output_dir, args.shap_data_name)
            torch.save(shap_data, shap_data_path)
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

            print(f"\nSHAP analysis data saved to '{shap_data_path}'")

            print("\n" + "="*70)
            print("SHAP ANALYSIS COMPLETE")
            print("Generated files:")
            print(f"   • {shap_plot_path} (visualizations)")
            print(f"   • {heatmap_save_path} (fragile sites heatmap)") 
            print(f"   • {curves_save_path} (fragile sites curves)")     
            print(f"   • {roc_save_path} (ROC/AUC curve)")              
            print(f"   • {shap_report_path} (detailed report)")
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