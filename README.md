# Antimicrobial Peptide Prediction with Transformer Models

This repository contains two complementary pipelines for predicting antimicrobial peptides (AMPs) using state-of-the-art transformer models:

1. **BERT Pipeline**: Uses BERT-base-uncased with character-level tokenization
2. **ESM-2 Pipeline**: Uses ESM-2 protein language model with native protein tokenization

## Project Overview

Antimicrobial peptides are crucial components of innate immunity with potential therapeutic applications. This project implements deep learning approaches to predict AMP activity from protein sequences, comparing general language models (BERT) with specialized protein language models (ESM-2).

## Repository Structure

```
transformer_model_pipelines/
├── BERT_Pipeline/
│   ├── bert_amp_pipeline.py    # Main BERT pipeline
│   ├── config.py               # Configuration management
│   ├── requirements.txt        # Dependencies
│   └── README.md              # BERT-specific documentation
├── ESM-2_Pipeline/
│   ├── esm2_amp_pipeline.py   # Main ESM-2 pipeline
│   ├── config.py              # Configuration management
│   ├── requirements.txt       # Dependencies
│   └── README.md             # ESM-2-specific documentation
└── README.md                 # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 50GB+ disk space for models and data

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zainsohail0/transformer_model_pipelines.git
cd transformer_model_pipelines
```

2. Choose your pipeline and install dependencies:

**For BERT Pipeline:**
```bash
cd BERT_Pipeline
pip install -r requirements.txt
```

**For ESM-2 Pipeline:**
```bash
cd ESM-2_Pipeline
pip install -r requirements.txt
```

### Data Preparation

Prepare your data in FASTA format:
- `positive_samples.fasta`: Confirmed antimicrobial peptides
- `negative_samples.fasta`: Non-antimicrobial peptides/proteins

Example structure:
```
data/
├── positive_samples.fasta
└── negative_samples.fasta
```

### Running the Pipelines

**BERT Pipeline:**
```bash
cd BERT_Pipeline
python bert_amp_pipeline.py --data_dir ../data --pos_fasta positive_samples.fasta --neg_fasta negative_samples.fasta
```

**ESM-2 Pipeline:**
```bash
cd ESM-2_Pipeline
python esm2_amp_pipeline.py --data_dir ../data --pos_fasta positive_samples.fasta --neg_fasta negative_samples.fasta
```

## Methodology

### BERT Pipeline
- **Tokenization**: Character-level with spaces between amino acids
- **Model**: BERT-base-uncased (110M parameters)  
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Max Length**: 160 tokens
- **Advantage**: Fast, well-established architecture

### ESM-2 Pipeline  
- **Tokenization**: Native protein tokenization (no spaces needed)
- **Model**: ESM-2-150M or ESM-2-650M parameters
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Max Length**: 160 amino acids
- **Advantage**: Protein-specific pre-training, better biological understanding

### Evaluation Strategy
- **Cross-validation**: Stratified 10-fold CV
- **Test Split**: 20% holdout set
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Early Stopping**: Prevents overfitting

## Expected Results

Both pipelines typically achieve:
- **Accuracy**: 85-95%
- **F1-Score**: 0.80-0.90
- **ROC-AUC**: 0.90-0.98

ESM-2 generally shows slightly better performance due to protein-specific pre-training.

## Tips for Best Results

1. **Data Quality**: Ensure your positive and negative samples are well-curated
2. **Balance**: Maintain reasonable class balance (recommend 1:1 to 1:3 ratio)
3. **Sequence Length**: Most AMPs are 10-50 amino acids; adjust `max_length` accordingly
4. **Hardware**: Use GPU for faster training (10-100x speedup)
5. **Hyperparameters**: Start with defaults, then tune learning rate and batch size

## Customization

### Configuration Options
Both pipelines support extensive configuration via:
- Command-line arguments
- Environment variables  
- Direct config file modification

### Key Parameters to Tune
- `learning_rate`: Start with 2e-5 (BERT) or 1e-5 (ESM-2)
- `batch_size`: Adjust based on GPU memory (16-64)
- `epochs`: Usually 3-10 epochs are sufficient
- `max_length`: Match your typical sequence lengths

### Model Variants
- **BERT**: Can use other BERT variants (RoBERTa, DistilBERT)
- **ESM-2**: Multiple sizes available (150M, 650M, 3B parameters)

## Output Files

Each pipeline generates:
- Model checkpoints (`.pt` files)
- Training metrics (CSV files)
- Final test results (JSON)
- Confusion matrices (CSV)
- Loss curves (PNG plots)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

**BERT:**
```bibtex
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

**ESM-2:**
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
