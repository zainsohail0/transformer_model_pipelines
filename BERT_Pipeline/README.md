# BERT Antimicrobial Peptide Prediction Pipeline

This pipeline implements a BERT-based model with LoRA fine-tuning for predicting antimicrobial peptides (AMPs) from protein sequences using stratified k-fold cross-validation.

## Overview

The pipeline uses:
- **BERT-base-uncased** as the base transformer model
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- **Stratified 10-fold cross-validation** for robust evaluation
- **Early stopping** to prevent overfitting

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

Your FASTA files should be in standard format:
```
>sequence_id_1
MKLLVLSLVLLLITGTQSQSAKGILQNLTSADLKNKQFNNTGKDLQAIQYLQKYGAKDIALETRQYEESGAFIR
>sequence_id_2
KLAKLAKKLAKLAK
```

## Usage

### Basic Usage
```bash
python bert_amp_pipeline.py --data_dir ./data --pos_fasta positive_samples.fasta --neg_fasta negative_samples.fasta
```

### Advanced Configuration
```bash
python bert_amp_pipeline.py \
    --data_dir ./data \
    --pos_fasta positive_samples.fasta \
    --neg_fasta negative_samples.fasta \
    --output_dir ./results \
    --batch_size 32 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --max_length 160
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Directory containing FASTA files |
| `--pos_fasta` | `test_pos_clean.fasta` | Positive samples FASTA file |
| `--neg_fasta` | `negatives_train.fasta` | Negative samples FASTA file |
| `--output_dir` | `./outputs` | Output directory |
| `--cache_dir` | `./cache` | Cache directory for processed data |
| `--checkpoint_dir` | `./checkpoints` | Model checkpoints directory |
| `--batch_size` | `32` | Training batch size |
| `--epochs` | `3` | Number of training epochs |
| `--learning_rate` | `2e-5` | Learning rate |
| `--max_length` | `160` | Maximum sequence length |
| `--n_splits` | `10` | Number of CV folds |
| `--test_size` | `0.2` | Test set proportion |

## Environment Variables

You can also set configuration via environment variables:
```bash
export DATA_DIR=/path/to/data
export POS_FASTA=positive.fasta
export NEG_FASTA=negative.fasta
export OUTPUT_DIR=/path/to/outputs
```

## Output Files

The pipeline generates:
- `checkpoints/fold{N}_best.pt` - Best model weights for each fold
- `checkpoints/metrics/fold{N}_epoch_metrics.csv` - Per-epoch metrics
- `checkpoints/metrics/test_metrics.json` - Final test set results
- `checkpoints/metrics/confusion_matrix.csv` - Confusion matrix
- `checkpoints/val_loss_curve_all_folds.png` - Validation loss plot

## Model Architecture

- **Base Model**: BERT-base-uncased
- **Fine-tuning**: LoRA with rank=8, alpha=16
- **Target Modules**: Query and Value attention layers
- **Classification Head**: Binary classification (AMP vs non-AMP)

## Citation

If you use this pipeline in your research, please cite the original BERT paper:
```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```