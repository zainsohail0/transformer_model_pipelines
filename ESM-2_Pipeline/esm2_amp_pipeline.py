#!/usr/bin/env python3
"""
ESM-2 Antimicrobial Peptide Prediction Pipeline

This script implements an ESM-2-based model with LoRA fine-tuning for predicting
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
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW

from config import get_config

class HuggingESM2PlusMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        base_model = EsmForSequenceClassification.from_pretrained(
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
    """Create ESM-2 tokenizer - no need to space out amino acids like with BERT"""
    return EsmTokenizer.from_pretrained(config.tokenizer_name)

def encode_batch(sequences, max_len=160):
    input_ids, attention_masks, labels = [], [], []
    for seq, label in sequences:
        # ESM-2 expects raw protein sequences (no spacing needed)
        encoded = tokenizer(
            seq,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(label)
    
    return input_ids, attention_masks, labels


import os
cached_path = "/scratch/negishi/sohailz/esmFiles/encoded_dataset.pt"  # New path for ESM data
if os.path.exists(cached_path):
    print("Loading cached tokenized ESM-2 dataset...")
    dataset = torch.load(cached_path)
else:
    print("Tokenizing from FASTA for ESM-2...")  
    # Load FASTA files
    pos_data = load_fasta("/scratch/negishi/sohailz/bertFiles/test_pos_clean.fasta", 1)
    neg_data = load_fasta("/scratch/negishi/sohailz/bertFiles/negatives_train.fasta", 0)
    
    # Combine and encode
    all_data = pos_data + neg_data
    input_ids, attention_masks, labels = encode_batch(all_data)
    
    # Package into a dataset
    dataset = TensorDataset(torch.stack(input_ids),
                            torch.stack(attention_masks),
                            torch.tensor(labels))
    
    print("Saving encoded ESM-2 dataset...")
    torch.save(dataset, cached_path)

import numpy as np
all_labels = np.array([dataset[i][2].item() for i in range(len(dataset))])
train_val_idx, test_idx = train_test_split(
    list(range(len(dataset))), test_size=0.2, stratify=all_labels, random_state=42, shuffle=True
)
train_val_dataset = Subset(dataset, train_val_idx)
test_dataset = Subset(dataset, test_idx)
train_val_labels = all_labels[train_val_idx]
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10
accumulation_steps = 4
batch_size = 32  # ESM-2 is larger, so smaller batch size
patience = 2
checkpoint_dir = "/scratch/negishi/sohailz/esmFiles/checkpointsESM"  # New checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)
loss_path = os.path.join(checkpoint_dir, "fold_losses.pt")
metrics_dir = os.path.join(checkpoint_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

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


from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
import torch, os, csv, json
import matplotlib.pyplot as plt
import pandas as pd
EPOCHS = 3

# Load ESM-2 pretrained weights if you have them
# ckpt = torch.load("/path/to/esm2/checkpoint.pt", map_location=device)  # Update path as needed

class_weights = torch.tensor([1.0, 1.0]).to(device)

print(f"Starting / resuming ESM-2 training at fold {start_fold}")

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset, train_val_labels), start=1):
    if fold < start_fold:
        print(f"Skipping fold {fold} (already completed)")
        continue

    print(f"\n━━━━━━━━━━━━━━━━  ESM-2 FOLD {fold}  ━━━━━━━━━━━━━━━━")
    train_loader = DataLoader(Subset(train_val_dataset, train_idx), 
                              batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(Subset(train_val_dataset, val_idx), 
                            batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, prefetch_factor=4)

    model = HuggingESM2PlusMLP().to(device)
    
    # If you have ESM-2 pretrained weights, load them here:
    # model.model.base_model.load_state_dict(ckpt, strict=False)
    
    # Freeze base model and enable adapters
    model.model.base_model.requires_grad_(False)
    model.model.enable_adapter_layers()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)  # Lower LR for ESM-2
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"ESM-2 Fold {fold} • Epoch {epoch} • train")):
            ids, masks, labels = [x.to(device, non_blocking=True) for x in batch]
            loss = loss_fn(model(ids, attention_mask=masks), labels)
            loss.backward()

            if (step+1) % accumulation_steps == 0 or step == len(train_loader)-1:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── VALIDATION ─
        model.eval()
        val_sum = 0.0
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"ESM-2 Fold {fold} • Epoch {epoch} • val"):
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

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
              f"Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | "
              f"F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        # Save epoch metrics
        fold_metrics_csv = os.path.join(metrics_dir, f"fold{fold}_epoch_metrics.csv")
        write_header = not os.path.exists(fold_metrics_csv)
        with open(fold_metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "roc_auc"])
            writer.writerow([epoch, avg_train, avg_val, val_acc, val_prec, val_rec, val_f1, val_auc])

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"fold{fold}_best.pt"))
            print("   -> New best ESM-2 model saved for this fold")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("   ↳ Early-stopping for this fold")
                break

    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)

    # Save progress
    torch.save({
        "last_completed_fold": fold,
        "fold_train_losses": fold_train_losses,
        "fold_val_losses": fold_val_losses,
    }, loss_path)

# Plotting code (same as before)
plt.figure(figsize=(10, 5))
for i, vl in enumerate(fold_val_losses, start=1):
    plt.plot(vl, label=f"Fold {i}")
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("ESM-2 Validation Loss per Fold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "esm2_val_loss_curve_all_folds.png"))
plt.close()

# Combined metrics plotting
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
    plt.title("ESM-2 Validation Log Loss per Epoch (per Fold)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Log Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "esm2_log_loss_per_epoch.png"))
    plt.close()

print("Saved ESM-2 combined loss plot ->", os.path.join(checkpoint_dir, "esm2_val_loss_curve_all_folds.png"))



from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, log_loss)
import json, csv, os
from tqdm import tqdm

# ── 1. Locate the best fold (lowest val-loss ever seen) ────────────────────────────────
best_fold, best_val = None, float("inf")
for fold_idx, losses in enumerate(fold_val_losses, start=1):
    fold_min = min(losses)
    if fold_min < best_val:
        best_val  = fold_min
        best_fold = fold_idx

best_ckpt = os.path.join(checkpoint_dir, f"fold{best_fold}_best.pt")
print(f"Best ESM-2 fold = {best_fold}  (min val-loss {best_val:.4f})  ->  {best_ckpt}")

# ── 2. Load model and weights correctly ────────────────────────────────────────────────
model = HuggingESM2PlusMLP().to(device)
model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=True)
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
    for ids, masks, labels in tqdm(test_loader, desc="ESM-2 Test-set"):
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
cm   = confusion_matrix(all_labels, all_preds)

print(f"\n── ESM-2 Test-set results ──\n"
      f"Accuracy  : {acc:.4f}\nPrecision : {prec:.4f}\nRecall    : {rec:.4f}\n"
      f"F1-score  : {f1:.4f}\nROC-AUC   : {auc:.4f}\nLog-loss  : {ll:.4f}")

# ── 6. Persist results ─────────────────────────────────────────────────────────────────
# 6a. Confusion-matrix CSV
cm_path = os.path.join(metrics_dir, "esm2_confusion_matrix.csv")
with open(cm_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["", "Pred 0", "Pred 1"])
    writer.writerow(["True 0", cm[0,0], cm[0,1]])
    writer.writerow(["True 1", cm[1,0], cm[1,1]])

# 6b. JSON summary of scalar metrics
metrics_path = os.path.join(metrics_dir, "esm2_test_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({
        "model_type" : "ESM-2",
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

print(f"\nSaved ESM-2 metrics -> {metrics_path}")
print(f"Saved ESM-2 confusion-matrix CSV -> {cm_path}")

if __name__ == "__main__":
    main()