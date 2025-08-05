# ESM-2 Antimicrobial Peptide Prediction Pipeline

This pipeline implements an ESM-2 (Evolutionary Scale Modeling) based model with LoRA fine-tuning for predicting antimicrobial peptides (AMPs) from protein sequences using stratified k-fold cross-validation.

## Overview

The pipeline uses:
- **ESM-2 (facebook/esm2_t30_150M_UR50D)** as the base protein language model
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
python esm2_amp_pipeline.py --data_dir ./data --pos_fasta positive_samples.fasta --neg_fasta negative_samples.fasta
```

### Advanced Configuration
```bash
python esm2_amp_pipeline.py \
    --data_dir ./data \
    --pos_fasta positive_samples.fasta \
    --neg_fasta negative_samples.fasta \
    --output_dir ./results \
    --batch_size 16 \
    --epochs 5 \
    --learning_rate 1e-5 \
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
| `--model_name` | `facebook/esm2_t30_150M_UR50D` | ESM-2 model variant |
| `--batch_size` | `32` | Training batch size (use smaller for larger models) |
| `--epochs` | `3` | Number of training epochs |
| `--learning_rate` | `1e-5` | Learning rate (lower than BERT) |
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
- `checkpoints/metrics/esm2_test_metrics.json` - Final test set results
- `checkpoints/metrics/esm2_confusion_matrix.csv` - Confusion matrix
- `checkpoints/esm2_val_loss_curve_all_folds.png` - Validation loss plot

## Model Architecture

- **Base Model**: ESM-2 (Evolutionary Scale Modeling v2)
- **Model Size**: 150M parameters (esm2_t30_150M_UR50D)
- **Fine-tuning**: LoRA with rank=8, alpha=16
- **Target Modules**: Query and Value attention layers
- **Classification Head**: Binary classification (AMP vs non-AMP)

## ESM-2 Model Variants

You can choose different ESM-2 model sizes:
- `facebook/esm2_t30_150M_UR50D` - 150M parameters (default, faster)
- `facebook/esm2_t33_650M_UR50D` - 650M parameters (slower, potentially better)
- `facebook/esm2_t36_3B_UR50D` - 3B parameters (requires significant GPU memory)

## Citation

If you use this pipeline in your research, please cite the ESM-2 paper:
```
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yossi and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```