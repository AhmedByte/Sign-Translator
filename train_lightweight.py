import argparse
import csv
import json
import os
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────────────────────────────────────────────────────
# Lightweight LSTM Model
# ───────────────────────────────────────────────────────────────────────────
class SignLSTM(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, num_layers=2, num_classes=10):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, seq_lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed_x)
        out = self.fc(hidden[-1])
        return out

# ───────────────────────────────────────────────────────────────────────────
# Dataset and Loader
# ───────────────────────────────────────────────────────────────────────────
class SignDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        if not self.augment:
            return feats, label
            
        feats = feats.clone()
        T, num_feats = feats.shape
        coord_dim = num_feats // 2
        
        # 1. Random Gaussian Noise
        if random.random() < 0.5:
            noise = torch.randn_like(feats) * 0.003
            feats = feats + noise
            
        # 2. Random Spatial Shift
        x_coords = feats[:, :coord_dim]
        y_coords = feats[:, coord_dim:]
        if random.random() < 0.5:
            shift_x = random.uniform(-0.03, 0.03)
            shift_y = random.uniform(-0.03, 0.03)
            x_coords = x_coords + shift_x
            y_coords = y_coords + shift_y
            
        x_coords = torch.clamp(x_coords, 0.0, 1.0)
        y_coords = torch.clamp(y_coords, 0.0, 1.0)
        return torch.cat([x_coords, y_coords], dim=-1), label

def collate_fn(batch):
    feats, labels = zip(*batch)
    seq_lengths = torch.tensor([len(f) for f in feats])
    padded_feats = nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_feats, seq_lengths, labels_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_extracted", help="Path to cached extracted feature files")
    parser.add_argument("--labels_name", default="labels_all.csv", help="Labels CSV file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_model", default="model_lightweight.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Using Device : {device.upper()}")

    # 1. Read labels
    labels_file = Path(args.labels_name)
    valid_class_ids = set()
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_class_ids.add(row["id"].strip())
    
    # 2. Load Cached Data
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache directory {args.cache_dir} not found. Run extract script first or train_custom.py")
        return

    # Find all .pt files in cache
    all_files = list(cache_path.glob("*.pt"))
    class_to_samples = {}
    
    for f in all_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            class_id = parts[0]
            if class_id in valid_class_ids:
                if class_id not in class_to_samples:
                    class_to_samples[class_id] = []
                class_to_samples[class_id].append(f)

    if not class_to_samples:
        print("❌ No matching cached data found.")
        return

    # Create mapping
    sorted_classes = sorted(list(class_to_samples.keys()))
    num_classes = len(sorted_classes)
    print(f"📂 Found {num_classes} classes.")

    train_samples = []
    test_samples = []

    for label_idx, class_id in enumerate(sorted_classes):
        files = sorted(class_to_samples[class_id])
        if len(files) < 3:
            train_files = files
            test_files = files
        else:
            train_files = files[:-1]
            test_files = files[-1:]

        for tf in train_files:
            feats = torch.load(tf, map_location="cpu")
            train_samples.append((feats, label_idx))
        for tf in test_files:
            feats = torch.load(tf, map_location="cpu")
            test_samples.append((feats, label_idx))

    print(f"📊 Train Samples : {len(train_samples)}")
    print(f"📊 Test Samples  : {len(test_samples)}")

    train_loader = DataLoader(SignDataset(train_samples, augment=True), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(SignDataset(test_samples, augment=False), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model, Optimizer, Loss
    model = SignLSTM(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("\n⏳ Training Lightweight LSTM ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for feats, seq_lengths, labels in train_loader:
            feats, seq_lengths, labels = feats.to(device), seq_lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats, seq_lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            _, preds = torch.max(logits, dim=-1)
            correct_train += (preds == labels).sum().item()
            total_train += len(labels)

        train_acc = (correct_train / total_train) * 100
        
        model.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for feats, seq_lengths, labels in test_loader:
                feats, seq_lengths, labels = feats.to(device), seq_lengths.to(device), labels.to(device)
                logits = model(feats, seq_lengths)
                loss = criterion(logits, labels)
                test_loss += loss.item() * len(labels)
                _, preds = torch.max(logits, dim=-1)
                correct_test += (preds == labels).sum().item()
                total_test += len(labels)

        test_acc = (correct_test / total_test) * 100
        
        if epoch % 50 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.output_model)

    print(f"🎉 Training Completed! Best Test Accuracy: {best_acc:.1f}%")
    print(f"💾 Checkpoint saved to: {args.output_model}")

if __name__ == "__main__":
    main()
