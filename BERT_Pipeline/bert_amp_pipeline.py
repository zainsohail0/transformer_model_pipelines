#!/usr/bin/env python3
"""
BERT Antimicrobial Peptide Prediction Pipeline

This script implements a BERT-based model with LoRA fine-tuning for predicting
antimicrobial peptides from protein sequences using cross-validation.
"""

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import json
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, log_loss
)

from transformers import AutoModelForSequenceClassification, BertTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

from config import get_config

class HuggingBertPlusMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=2
        )
        lora_config = LoraConfig(
            r=config.lora_r, 
            lora_alpha=config.lora_alpha, 
            target_modules=["query", "value"],
            lora_dropout=config.lora_dropout, 
            bias="none", 
            task_type=TaskType.SEQ_CLS
        )
        self.model = get_peft_model(base_model, lora_config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_fasta(path, label):
    sequences = []
    with open(path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                sequences.append((line.strip(), label))
    return sequences

def create_tokenizer(config):
    return BertTokenizer.from_pretrained(config.model_name)

def encode_batch(sequences, tokenizer, max_len=160):
    input_ids, attention_masks, labels = [], [], []
    for seq, label in sequences:
        spaced_seq = " ".join(list(seq))
        tokens = tokenizer.tokenize(spaced_seq)[:max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(ids)
        padding_length = max_len - len(ids)
        ids += [0] * padding_length
        mask += [0] * padding_length
        input_ids.append(torch.tensor(ids))
        attention_masks.append(torch.tensor(mask))
        labels.append(label)
    return input_ids, attention_masks, labels

def load_and_process_data(config, tokenizer):
    """Load FASTA files and create tokenized dataset."""
    cached_path = os.path.join(config.cache_dir, "encoded_dataset.pt")
    
    if os.path.exists(cached_path):
        print("Loading cached tokenized dataset...")
        dataset = torch.load(cached_path)
    else:
        print("Tokenizing from FASTA...")  
        
        # Load FASTA files
        pos_path = os.path.join(config.data_dir, config.pos_fasta)
        neg_path = os.path.join(config.data_dir, config.neg_fasta)
        
        pos_data = load_fasta(pos_path, 1)
        neg_data = load_fasta(neg_path, 0)
        
        # Combine and encode
        all_data = pos_data + neg_data
        input_ids, attention_masks, labels = encode_batch(all_data, tokenizer, config.max_length)
        
        # Package into a dataset
        dataset = TensorDataset(torch.stack(input_ids),
                                torch.stack(attention_masks),
                                torch.tensor(labels))
        
        print("Saving encoded dataset...")
        torch.save(dataset, cached_path)
    
    return dataset

def main():
    # Get configuration
    config = get_config()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer and load data
    tokenizer = create_tokenizer(config)
    dataset = load_and_process_data(config, tokenizer)
    
    # Split data
    all_labels = np.array([dataset[i][2].item() for i in range(len(dataset))])
    train_val_idx, test_idx = train_test_split(
        list(range(len(dataset))), 
        test_size=config.test_size, 
        stratify=all_labels, 
        random_state=config.random_state, 
        shuffle=True
    )
    
    train_val_dataset = Subset(dataset, train_val_idx)
    test_dataset = Subset(dataset, test_idx)
    train_val_labels = all_labels[train_val_idx]
    
    kf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    
    # Setup paths
    loss_path = os.path.join(config.checkpoint_dir, "fold_losses.pt")
    metrics_dir = os.path.join(config.checkpoint_dir, "metrics")
    
    # Resume info
    if os.path.exists(loss_path):
        state = torch.load(loss_path)
        start_fold = state["last_completed_fold"] + 1
        fold_train_losses = state["fold_train_losses"]
        fold_val_losses = state["fold_val_losses"]
    else:
        start_fold = 1
        fold_train_losses = []
        fold_val_losses = []

    # Setup class weights and pretrained checkpoint (if available)
    class_weights = torch.tensor([1.0, 1.0]).to(device)
    
    # Try to load pretrained BERT checkpoint
    pretrained_path = os.path.join(config.checkpoint_dir, "bert_pretrained.bin")
    state_dict = None
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained checkpoint from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)
        state_dict = ckpt.get("state_dict") or ckpt.get("params")
        if state_dict is None:
            print("Warning: Checkpoint missing state_dict or params, using default initialization")
    else:
        print("No pretrained checkpoint found, using default initialization")
    
    print(f"Starting / resuming at fold {start_fold}")

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset, train_val_labels), start=1):
        if fold < start_fold:
            print(f"Skipping fold {fold} (already completed)")
            continue

        print(f"\n{'='*50} FOLD {fold} {'='*50}")
        train_loader = DataLoader(
            Subset(train_val_dataset, train_idx), 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            pin_memory=config.pin_memory, 
            prefetch_factor=config.prefetch_factor
        )
        val_loader = DataLoader(
            Subset(train_val_dataset, val_idx), 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=config.pin_memory, 
            prefetch_factor=config.prefetch_factor
        )

        model = HuggingBertPlusMLP(config).to(device)
        if state_dict is not None:
            model.model.load_state_dict(state_dict, strict=False)
        model.model.base_model.requires_grad_(False)
        model.model.enable_adapter_layers()

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        patience_ctr = 0
        train_losses, val_losses = [], []

        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            
            for step, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold} • Epoch {epoch} • train")):
                ids, masks, labels = [x.to(device, non_blocking=True) for x in batch]
                loss = loss_fn(model(ids, attention_mask=masks), labels)
                loss.backward()

                if (step+1) % config.accumulation_steps == 0 or step == len(train_loader)-1:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── VALIDATION ─
        model.eval(); val_sum = 0.0
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold} • Epoch {epoch} • val"):
                ids, masks, labels = [x.to(device, non_blocking=True) for x in batch]
                logits = model(ids, attention_mask=masks)
                val_sum += loss_fn(logits, labels).item()
                probs = torch.softmax(logits, dim=1)
                val_probs.extend(probs[:,1].cpu().numpy())
                val_preds.extend(torch.argmax(probs, 1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val = val_sum / len(val_loader)
        val_losses.append(avg_val)

        val_acc  = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds, zero_division=0)
        val_rec  = recall_score(val_labels, val_preds, zero_division=0)
        val_f1   = f1_score(val_labels, val_preds, zero_division=0)
        val_auc  = roc_auc_score(val_labels, val_probs)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        fold_metrics_csv = os.path.join(metrics_dir, f"fold{fold}_epoch_metrics.csv")
        write_header = not os.path.exists(fold_metrics_csv)
        with open(fold_metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "roc_auc"])
            writer.writerow([epoch, avg_train, avg_val, val_acc, val_prec, val_rec, val_f1, val_auc])

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"fold{fold}_best.pt"))
            print("   -> New best model saved for this fold")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("   ↳ Early-stopping for this fold")
                break

    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)

    torch.save({
        "last_completed_fold": fold,
        "fold_train_losses": fold_train_losses,
        "fold_val_losses": fold_val_losses,
    }, loss_path)

plt.figure(figsize=(10, 5))
for i, vl in enumerate(fold_val_losses, start=1):
    plt.plot(vl, label=f"Fold {i}")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("Validation Loss per Fold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "val_loss_curve_all_folds.png"))
plt.close()

epoch_data = []
for fold in range(1, len(fold_val_losses) + 1):
    csv_path = os.path.join(metrics_dir, f"fold{fold}_epoch_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["fold"] = fold
        epoch_data.append(df)

if epoch_data:
    combined_df = pd.concat(epoch_data, ignore_index=True)
    plt.figure(figsize=(10, 6))
    for fold in range(1, len(fold_val_losses) + 1):
        df_fold = combined_df[combined_df["fold"] == fold]
        plt.plot(df_fold["epoch"], df_fold["val_loss"], label=f"Fold {fold}")
    plt.title("Validation Log Loss per Epoch (per Fold)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "log_loss_per_epoch.png"))
    plt.close()

print("Saved combined loss plot ->", os.path.join(checkpoint_dir, "val_loss_curve_all_folds.png"))

# ──────────────────────────── Cell 6 – Test-set evaluation ─────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, log_loss)
import json, csv, os

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
import torch, os, csv, json
import matplotlib.pyplot as plt
import pandas as pd

# ── 1. Locate the best fold (lowest val-loss ever seen) ────────────────────────────────
best_fold, best_val = None, float("inf")
for fold_idx, losses in enumerate(fold_val_losses, start=1):
    fold_min = min(losses)
    if fold_min < best_val:
        best_val  = fold_min
        best_fold = fold_idx

best_ckpt = os.path.join(checkpoint_dir, f"fold{best_fold}_best.pt")
print(f"Best fold = {best_fold}  (min val-loss {best_val:.4f})  ->  {best_ckpt}")

# ── 2. Load model and weights correctly ────────────────────────────────────────────────
model = HuggingBertPlusMLP().to(device)
model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=True)  # full fine-tuned model
model.eval()

# ── 3. Prepare test loader ─────────────────────────────────────────────────────────────
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True,
                         prefetch_factor=4)

# ── 4. Run inference ──────────────────────────────────────────────────────────────────
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for ids, masks, labels in tqdm(test_loader, desc="Test-set"):
        ids, masks = ids.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(ids, attention_mask=masks)
        probs  = torch.softmax(logits, dim=1)[:, 1]          # P(class = 1)
        preds  = torch.argmax(logits, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ── 5. Metrics ─────────────────────────────────────────────────────────────────────────
acc  = accuracy_score (all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec  = recall_score   (all_labels, all_preds, zero_division=0)
f1   = f1_score       (all_labels, all_preds, zero_division=0)
auc  = roc_auc_score  (all_labels, all_probs)
ll   = log_loss       (all_labels, all_probs)

cm   = confusion_matrix(all_labels, all_preds)  # 2×2 matrix: [[TN, FP], [FN, TP]]

print(f"\n── Test-set results ──\n"
      f"Accuracy  : {acc:.4f}\nPrecision : {prec:.4f}\nRecall    : {rec:.4f}\n"
      f"F1-score  : {f1:.4f}\nROC-AUC   : {auc:.4f}\nLog-loss  : {ll:.4f}")

# ── 6. Persist results ─────────────────────────────────────────────────────────────────
# 6a. Confusion-matrix CSV
cm_path = os.path.join(metrics_dir, "confusion_matrix.csv")
with open(cm_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["", "Pred 0", "Pred 1"])
    writer.writerow(["True 0", cm[0,0], cm[0,1]])
    writer.writerow(["True 1", cm[1,0], cm[1,1]])

# 6b. JSON summary of scalar metrics
metrics_path = os.path.join(metrics_dir, "test_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({
        "best_fold"  : best_fold,
        "val_loss"   : best_val,
        "accuracy"   : acc,
        "precision"  : prec,
        "recall"     : rec,
        "f1"         : f1,
        "roc_auc"    : auc,
        "log_loss"   : ll,
        "confusion_matrix": cm.tolist()
    }, f, indent=2)

print(f"\nSaved metrics -> {metrics_path}")
print(f"Saved confusion-matrix CSV -> {cm_path}")

if __name__ == "__main__":
    main()
