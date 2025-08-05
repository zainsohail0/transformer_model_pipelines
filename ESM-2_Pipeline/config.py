"""Configuration file for ESM-2 antimicrobial peptide prediction pipeline."""

import os
import argparse

def get_config():
    """Get configuration parameters from command line or environment variables."""
    parser = argparse.ArgumentParser(description='ESM-2 Antimicrobial Peptide Prediction Pipeline')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, 
                       default=os.getenv('DATA_DIR', './data'),
                       help='Directory containing FASTA files')
    parser.add_argument('--pos_fasta', type=str,
                       default=os.getenv('POS_FASTA', 'test_pos_clean.fasta'),
                       help='Positive samples FASTA file')
    parser.add_argument('--neg_fasta', type=str,
                       default=os.getenv('NEG_FASTA', 'negatives_train.fasta'),
                       help='Negative samples FASTA file')
    
    # Output paths
    parser.add_argument('--output_dir', type=str,
                       default=os.getenv('OUTPUT_DIR', './outputs'),
                       help='Directory for saving outputs')
    parser.add_argument('--cache_dir', type=str,
                       default=os.getenv('CACHE_DIR', './cache'),
                       help='Directory for caching processed datasets')
    parser.add_argument('--checkpoint_dir', type=str,
                       default=os.getenv('CHECKPOINT_DIR', './checkpoints'),
                       help='Directory for saving model checkpoints')
    
    # Model parameters
    parser.add_argument('--model_name', type=str,
                       default='facebook/esm2_t30_150M_UR50D',
                       help='ESM-2 model name')
    parser.add_argument('--tokenizer_name', type=str,
                       default='facebook/esm2_t33_650M_UR50D',
                       help='ESM-2 tokenizer name')
    parser.add_argument('--max_length', type=int, default=160,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=2,
                       help='Early stopping patience')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='Pin memory for data loader')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                       help='Prefetch factor for data loader')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'metrics'), exist_ok=True)
    
    return args