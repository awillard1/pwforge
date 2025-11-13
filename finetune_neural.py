#!/usr/bin/env python3
"""
Fine-tune a character-level LSTM for password generation.
Works on any .txt or .gz file (rockyou.txt, custom leaks, etc.).
"""

import argparse, gzip, os, sys, math, random
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# Config – printable ASCII 33-126
# -------------------------------------------------
PRINTABLE = ''.join(chr(i) for i in range(33, 127))   # ! " # $ ... ~
CHAR_TO_IDX = {c: i for i, c in enumerate(PRINTABLE)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
VOCAB_SIZE = len(PRINTABLE)

# BOS = '!'  (inside printable range)
# EOS = '\n' (fallback to index 0 if missing)
BOS_CHAR = '!'
EOS_CHAR = '\n'
BOS_IDX = CHAR_TO_IDX[BOS_CHAR]
EOS_IDX = CHAR_TO_IDX.get(EOS_CHAR, 0)   # <-- safe fallback

# -------------------------------------------------
# Dataset
# -------------------------------------------------
class PasswordDataset(Dataset):
    def __init__(self, lines, max_len=32):
        self.data = []
        for line in lines:
            pw = line.strip()
            if not pw or len(pw) < 3 or len(pw) > max_len:
                continue
            seq = [BOS_IDX] + [CHAR_TO_IDX.get(c, 0) for c in pw] + [EOS_IDX]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]          # input, target (shifted)

def collate_batch(batch):
    inputs = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    input_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return input_padded, target_padded

# -------------------------------------------------
# Model
# -------------------------------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=384,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")

    # ----- Load data -----
    print(f"[i] Loading passwords from {args.input}...")
    opener = gzip.open if args.input.endswith(('.gz', '.gzip')) else open
    lines = []
    with opener(args.input, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lines.append(line)
            if len(lines) % 1_000_000 == 0:
                print(f"    Loaded {len(lines):,} lines...")

    print(f"[i] Total lines: {len(lines):,}")
    random.shuffle(lines)

    split_idx = int(len(lines) * 0.95)
    train_lines = lines[:split_idx]
    val_lines   = lines[split_idx:]

    train_ds = PasswordDataset(train_lines, max_len=args.max_len)
    val_ds   = PasswordDataset(val_lines,   max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_batch)

    print(f"[i] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ----- Model -----
    model = CharLSTM(vocab_size=VOCAB_SIZE,
                     embed_dim=args.embed_dim,
                     hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers,
                     dropout=args.dropout).to(device)

    if args.pretrained:
        print(f"[i] Loading pretrained model: {args.pretrained}")
        pretrained = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(pretrained if isinstance(pretrained, dict) else pretrained.state_dict())

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    patience=2, factor=0.5)

    print(f"[i] Training for {args.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train = total_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), args.output)
            print(f"    [Saved] {args.output}")

    print(f"[i] Fine-tuning complete. Best model → {args.output}")

# -------------------------------------------------
# Sampling test (optional)
# -------------------------------------------------
@torch.no_grad()
def sample_model(model, device, count=10, max_len=32):
    model.eval()
    results = []
    for _ in range(count):
        seq = torch.tensor([[BOS_IDX]], dtype=torch.long).to(device)
        for _ in range(max_len):
            logits, _ = model(seq)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1)
            if nxt.item() == EOS_IDX:
                break
            seq = torch.cat([seq, nxt], dim=1)
        pw = ''.join(IDX_TO_CHAR.get(i.item(), '') for i in seq[0, 1:])
        results.append(pw)
    return results

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune neural password model on any .txt/.gz file")
    ap.add_argument("input", help="Path to password list (.txt or .gz)")
    ap.add_argument("--output", default="finetuned_model.pt",
                    help="Output .pt file")
    ap.add_argument("--pretrained", help="Start from existing .pt model")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--hidden-dim", type=int, default=384)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=32,
                    help="Max password length to train on")
    ap.add_argument("--test", action="store_true",
                    help="Generate 10 samples after training")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] File not found: {args.input}")
        sys.exit(1)

    train_model(args)

    if args.test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CharLSTM(embed_dim=args.embed_dim,
                         hidden_dim=args.hidden_dim,
                         num_layers=args.num_layers,
                         dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(args.output, map_location=device))
        samples = sample_model(model, device, count=10)
        print("\n[i] Sample passwords:")
        for s in samples:
            print(f"    {s}")

if __name__ == "__main__":
    main()