"""
Task 1: News Topic Classifier Using BERT
DevelopersHub Corporation – AI/ML Engineering Internship

Fine-tunes bert-base-uncased on the AG News dataset for 4-class topic classification.
Classes: World (0), Sports (1), Business (2), Sci/Tech (3)
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 32
EPOCHS       = 3
LR           = 2e-5
TRAIN_SUBSET = 8000   # use a subset for faster training (full = 120 000)
TEST_SUBSET  = 2000
SAVE_PATH    = "bert_ag_news.pt"
LABEL_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────
#  1. Dataset
# ─────────────────────────────────────────
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data():
    print("Loading AG News dataset…")
    ds = load_dataset("ag_news")

    # Subsample for speed
    train_df = pd.DataFrame(ds["train"]).sample(TRAIN_SUBSET, random_state=42)
    test_df  = pd.DataFrame(ds["test"]).sample(TEST_SUBSET,  random_state=42)

    # Labels are 1-indexed in AG News → shift to 0-indexed
    train_texts  = train_df["text"].tolist()
    train_labels = (train_df["label"]).tolist()
    test_texts   = test_df["text"].tolist()
    test_labels  = (test_df["label"]).tolist()

    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    print("Label distribution (train):")
    print(pd.Series(train_labels).value_counts().rename(index=dict(enumerate(LABEL_NAMES))))
    return train_texts, train_labels, test_texts, test_labels


# ─────────────────────────────────────────
#  2. Training loop
# ─────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds_all.extend(out.logits.argmax(-1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted")
    return total_loss / len(loader), acc, f1


def eval_epoch(model, loader):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += out.loss.item()
            preds_all.extend(out.logits.argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted")
    return total_loss / len(loader), acc, f1, preds_all, labels_all


# ─────────────────────────────────────────
#  3. Visualisations
# ─────────────────────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, key, title, color in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc"), ("train_f1", "val_f1")],
        ["Loss", "Accuracy", "F1 Score"],
        ["#e74c3c", "#2ecc71", "#3498db"],
    ):
        ax.plot(epochs, history[key[0]], "o--", color=color, label="Train", alpha=0.8)
        ax.plot(epochs, history[key[1]], "o-",  color=color, label="Val",   linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("BERT Fine-tuning on AG News", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: training_curves.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title("Confusion Matrix – AG News BERT", fontsize=13)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrix.png")


# ─────────────────────────────────────────
#  4. Main
# ─────────────────────────────────────────
def main():
    train_texts, train_labels, test_texts, test_labels = load_data()

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = AGNewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_ds  = AGNewsDataset(test_texts,  test_labels,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model = model.to(DEVICE)

    total_steps = len(train_loader) * EPOCHS
    optimizer   = AdamW(model.parameters(), lr=LR, eps=1e-8)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]}

    print(f"\n{'─'*60}")
    print(f"Fine-tuning {MODEL_NAME} for {EPOCHS} epochs")
    print(f"{'─'*60}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1             = train_epoch(model, train_loader, optimizer, scheduler)
        va_loss, va_acc, va_f1, preds, labels = eval_epoch(model, test_loader)

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"],
                        [tr_loss, va_loss, tr_acc, va_acc, tr_f1, va_f1]):
            history[k].append(v)

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"loss={tr_loss:.4f}/{va_loss:.4f}  "
              f"acc={tr_acc:.4f}/{va_acc:.4f}  "
              f"f1={tr_f1:.4f}/{va_f1:.4f}")

    # Final evaluation
    print(f"\n{'─'*60}")
    print("Final Classification Report:")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    # Save model
    torch.save({"model_state": model.state_dict(), "config": MODEL_NAME}, SAVE_PATH)
    tokenizer.save_pretrained("tokenizer/")
    print(f"Model saved to {SAVE_PATH}")

    # Plots
    plot_training_curves(history)
    plot_confusion_matrix(labels, preds)

    return model, tokenizer


if __name__ == "__main__":
    main()
