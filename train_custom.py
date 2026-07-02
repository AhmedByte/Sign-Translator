"""
train_custom.py

A custom script to fine-tune the SignBart model on skeletal landmark data.
Automatically handles MediaPipe landmark extraction, caching, dataset partitioning,
and lightweight training by freezing the BART core and training the projection/classification heads.

Usage:
    python3 train_custom.py --num_classes 10 --epochs 25
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sign_bart import SignBart, SignBartConfig


# ───────────────────────────────────────────────────────────────────────────
# MediaPipe Feature Extraction
# ───────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_frames(frames_dir: Path) -> torch.Tensor:
    """Extract MediaPipe Holistic landmarks (96-dim) from a directory of sorted frame images."""
    import cv2
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except (ImportError, AttributeError):
        raise ImportError("MediaPipe is required for feature extraction. Please install it with 'pip install mediapipe'")

    # Find and sort frame images chronologically
    img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]
    
    img_paths = sorted([
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in img_extensions
    ], key=natural_sort_key)

    if not img_paths:
        raise ValueError(f"No image frames found in {frames_dir}")

    features = []

    # Using static_image_mode=True is optimal for high-quality extraction from individual folders of images
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
        for idx, img_path in enumerate(img_paths):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # 1. Pose landmarks (11, 12, 13, 14, 15, 16)
            pose_x = [0.0] * 6
            pose_y = [0.0] * 6
            if results.pose_landmarks:
                for j_idx, joint in enumerate([11, 12, 13, 14, 15, 16]):
                    landmark = results.pose_landmarks.landmark[joint]
                    pose_x[j_idx] = landmark.x
                    pose_y[j_idx] = landmark.y

            # 2. Left Hand landmarks (21 landmarks)
            lh_x = [0.0] * 21
            lh_y = [0.0] * 21
            if results.left_hand_landmarks:
                for h_idx in range(21):
                    landmark = results.left_hand_landmarks.landmark[h_idx]
                    lh_x[h_idx] = landmark.x
                    lh_y[h_idx] = landmark.y

            # 3. Right Hand landmarks (21 landmarks)
            rh_x = [0.0] * 21
            rh_y = [0.0] * 21
            if results.right_hand_landmarks:
                for h_idx in range(21):
                    landmark = results.right_hand_landmarks.landmark[h_idx]
                    rh_x[h_idx] = landmark.x
                    rh_y[h_idx] = landmark.y

            all_x = pose_x + lh_x + rh_x
            all_y = pose_y + lh_y + rh_y
            features.append(all_x + all_y)

    if not features:
        # Fallback to zero frames if something goes completely wrong
        features = [[0.0] * 96]

    return torch.tensor(features, dtype=torch.float32)


# ───────────────────────────────────────────────────────────────────────────
# Dataset and Loader
# ───────────────────────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, samples: List[Tuple[torch.Tensor, int]], augment: bool = False):
        """
        Args:
            samples: List of (feature_tensor, label_id) tuples.
            augment: Whether to apply landmark coordinate data augmentations.
        """
        self.samples = samples
        self.augment = augment
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        if not self.augment:
            return feats, label
            
        # Clone sequence tensor to avoid mutating original cache in RAM
        feats = feats.clone()
        T, num_feats = feats.shape
        coord_dim = num_feats // 2 # 48
        
        # 1. Random Gaussian Noise (Minor coordinate jitter)
        if random.random() < 0.5:
            noise = torch.randn_like(feats) * 0.003
            feats = feats + noise
            
        x_coords = feats[:, :coord_dim]
        y_coords = feats[:, coord_dim:]
        
        # 2. Random Spatial Shift / Translation (Actor standing slightly to left/right/up/down)
        if random.random() < 0.5:
            shift_x = random.uniform(-0.03, 0.03)
            shift_y = random.uniform(-0.03, 0.03)
            x_coords = x_coords + shift_x
            y_coords = y_coords + shift_y
            
        # 3. Random Spatial Scaling / Zoom (Actor standing closer or further)
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            x_mean = x_coords[x_coords != 0].mean() if (x_coords != 0).any() else 0.5
            y_mean = y_coords[y_coords != 0].mean() if (y_coords != 0).any() else 0.5
            x_coords = (x_coords - x_mean) * scale + x_mean
            y_coords = (y_coords - y_mean) * scale + y_mean
            
        # 4. Temporal Resampling / Speed Augmentation (Speeding up or slowing down the sign)
        if random.random() < 0.5:
            orig_len = len(x_coords)
            if orig_len > 5:
                speed_factor = random.uniform(0.8, 1.2)
                new_len = int(orig_len * speed_factor)
                new_len = max(5, new_len)
                
                # Create resampled index array
                indices = torch.linspace(0, orig_len - 1, new_len).long()
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]
            
        # Clamp to normalized [0.0, 1.0] range
        x_coords = torch.clamp(x_coords, 0.0, 1.0)
        y_coords = torch.clamp(y_coords, 0.0, 1.0)
        
        augmented_feats = torch.cat([x_coords, y_coords], dim=-1)
        return augmented_feats, label


def collate_fn(batch):
    """Pad sequence lengths to maximum sequence length in the batch."""
    feats, labels = zip(*batch)

    # Pad along sequence dimension (T) -> output shape: (B, T_max, 96)
    padded_feats = nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)

    # Build attention mask (1 for real frames, 0 for padded frames)
    batch_size = len(feats)
    max_len = padded_feats.shape[1]
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, f in enumerate(feats):
        attention_mask[i, :len(f)] = 1

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return padded_feats, attention_mask, labels_tensor


# ───────────────────────────────────────────────────────────────────────────
# Training Logic
# ───────────────────────────────────────────────────────────────────────────

def sep(title="", w=70):
    if title:
        p = (w - len(title) - 2) // 2
        print("=" * p + f" {title} " + "=" * max(0, w - p - len(title) - 2))
    else:
        print("=" * w)


def main():
    parser = argparse.ArgumentParser(description="SignLanguage custom BART Fine-Tuning script")
    parser.add_argument("--data_dir", default="data", help="Path to raw frame directories (e.g. data/)")
    parser.add_argument("--cache_dir", default="data_extracted", help="Path to cache extracted feature files")
    parser.add_argument("--checkpoint_dir", default=".", help="Path to config.json and weights")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes to train on (max 50)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="Resume training from train_snapshot.pt if it exists")
    parser.add_argument("--freeze_bart", action="store_true", help="Freeze core BART layers, training only projection and classification head")
    parser.add_argument("--train_all", action="store_true", help="Train on all available instances per class (no holdout split)")
    parser.add_argument("--config_name", default="config.json", help="Model config file name (e.g. config.json)")
    parser.add_argument("--labels_name", default="labels.csv", help="Labels CSV file name (e.g. labels.csv)")
    parser.add_argument("--output_model", default="model_finetuned.pt", help="Filename of the saved best checkpoint")
    args = parser.parse_args()

    sep("SignBart Fine-Tuning Setup")

    # Set up Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"📡 Using Device : {device.upper()}")

    # 1. Discover Data & Partition
    data_path = Path(args.data_dir)
    cache_path = Path(args.cache_dir)
    cache_path.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f"❌ Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)

    # Read labels to filter out any directories that were cleaned
    labels_file = PROJECT_ROOT / args.labels_name
    valid_class_ids = set()
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                valid_class_ids.add(row["id"].strip())
    else:
        print(f"⚠️ Warning: {labels_file} not found. Training on all available folders.")

    # Read class subfolders (only numeric folders like 0001, 0002, etc.) and filter by labels.csv
    class_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and d.name.isdigit() and (not valid_class_ids or d.name in valid_class_ids)
    ])

    if not class_dirs:
        print(f"❌ Error: No numeric class subfolders found in '{args.data_dir}'. Expected folders like '0001', '0002'.")
        sys.exit(1)

    # Limit to user request classes
    num_classes = min(len(class_dirs), args.num_classes)
    selected_classes = class_dirs[:num_classes]
    print(f"📂 Found {len(class_dirs)} classes, selected the first {num_classes} classes for training.")

    # 2. Extract Landmarks (Caching to avoid repeating heavy processing)
    print("\n⏳ [1/4] Starting Landmark Feature Extraction (or loading from cache)...")
    train_samples = []
    test_samples = []

    for class_dir in selected_classes:
        class_name = class_dir.name
        # Class Label ID: mapping dynamically to index in selected_classes to handle non-contiguous classes/subsets
        label_id = selected_classes.index(class_dir)

        # Get all instance subfolders (e.g. 1, 2, 3, 10) sorted numerically
        instance_dirs = sorted(
            [d for d in class_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name
        )
        if len(instance_dirs) == 0:
            # Fallback: check if there are image frames directly inside class_dir
            img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            has_images = any(p.suffix.lower() in img_extensions for p in class_dir.iterdir())
            if has_images:
                instance_dirs = [class_dir]
            else:
                print(f"⚠️ Warning: Class {class_name} has no subfolders or image files. Skipping class.")
                continue

        # Dynamic Partition: train on all if specified OR if there are fewer than 3 instances
        if args.train_all or len(instance_dirs) < 3:
            if len(instance_dirs) < 3:
                print(f"ℹ️ Class {class_name} has only {len(instance_dirs)} instance(s). Using all for training & validation.")
            train_instances = instance_dirs
            test_instances = instance_dirs
        else:
            train_instances = instance_dirs[:-1]
            test_instances = instance_dirs[-1:]

        # Process training instances
        for inst_dir in train_instances:
            cache_file = cache_path / f"{class_name}_{inst_dir.name}.pt"
            if cache_file.exists():
                feats = torch.load(cache_file, map_location="cpu")
            else:
                print(f"  ⚡ Extracting: {class_name}/{inst_dir.name} ...")
                feats = extract_landmarks_from_frames(inst_dir)
                torch.save(feats, cache_file)
            train_samples.append((feats, label_id))

        # Process testing instance
        for inst_dir in test_instances:
            cache_file = cache_path / f"{class_name}_{inst_dir.name}.pt"
            if cache_file.exists():
                feats = torch.load(cache_file, map_location="cpu")
            else:
                print(f"  ⚡ Extracting: {class_name}/{inst_dir.name} ...")
                feats = extract_landmarks_from_frames(inst_dir)
                torch.save(feats, cache_file)
            test_samples.append((feats, label_id))

    print(f"✅ Landmark extraction completed!")
    print(f"   📊 Train Samples : {len(train_samples)}")
    print(f"   📊 Test Samples  : {len(test_samples)}")

    # Create Dataloaders with training data augmentation enabled
    train_dataset = SignDataset(train_samples, augment=True)
    test_dataset = SignDataset(test_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Load Pretrained model
    print("\n⏳ [2/4] Loading pretrained SignBart model …")
    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = checkpoint_dir / args.config_name
    with open(str(config_path), "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    ALLOWED = {
        "vocab_size", "d_model", "encoder_layers", "decoder_layers",
        "encoder_attention_heads", "decoder_attention_heads",
        "encoder_ffn_dim", "decoder_ffn_dim",
        "max_position_embeddings", "dropout", "attention_dropout",
        "activation_function", "pad_token_id", "bos_token_id",
        "eos_token_id", "decoder_start_token_id", "is_encoder_decoder",
        "scale_embedding", "use_cache",
    }
    kwargs = {k: v for k, v in raw_cfg.items() if k in ALLOWED}
    num_labels = len(raw_cfg.get("id2label", {}))
    coord_dim = raw_cfg.get("coord_dim", 48)
    config = SignBartConfig(
        coord_dim=coord_dim,
        num_labels=num_labels,
        **kwargs,
    )

    weights_path = str(checkpoint_dir / "model.safetensors")
    model = SignBart.from_safetensors(config, weights_path, device=device)

    # Freeze core BART weights conditionally
    trainable_params = []
    frozen_params_count = 0
    trainable_params_count = 0
 
    for name, param in model.named_parameters():
        if args.freeze_bart:
            if "projection" in name or "classification_head" in name:
                param.requires_grad = True
                trainable_params.append(param)
                trainable_params_count += param.numel()
            else:
                param.requires_grad = False
                frozen_params_count += param.numel()
        else:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_params_count += param.numel()

    print(f"🔒 Frozen parameters    : {frozen_params_count:,}")
    print(f"🔥 Trainable parameters : {trainable_params_count:,}")

    # 4. Optimizer and Loss
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 5. Checkpoint Resume / Snapshot Logic
    start_epoch = 1
    best_acc = 0.0
    snapshot_path = Path(f"train_snapshot_{Path(args.output_model).stem}.pt")

    if args.resume or snapshot_path.exists():
        if snapshot_path.exists():
            try:
                print(f"🔄 Found unfinished training snapshot: {snapshot_path}")
                snapshot = torch.load(snapshot_path, map_location=device)
                
                # Check config compatibility
                snap_num_classes = snapshot.get("num_classes", 0)
                if snap_num_classes == args.num_classes:
                    model.load_state_dict(snapshot["model_state_dict"])
                    optimizer.load_state_dict(snapshot["optimizer_state_dict"])
                    start_epoch = snapshot["epoch"] + 1
                    best_acc = snapshot["best_acc"]
                    print(f"✅ Successfully resumed from Epoch {snapshot['epoch']} (Best Validation Acc: {best_acc:.1f}%)!")
                else:
                    print(f"⚠️ Snapshot class count ({snap_num_classes}) is different from current run class count ({args.num_classes}). Starting fresh.")
            except Exception as snap_err:
                print(f"⚠️ Error loading snapshot ({snap_err}). Starting training from scratch.")
        else:
            print("ℹ️ Resume flag set but train_snapshot.pt not found. Starting from scratch.")

    print("\n⏳ [3/4] Training Loop started ...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats, masks, labels in train_loader:
            feats = feats.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs_embeds=feats, attention_mask=masks)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            _, preds = torch.max(logits, dim=-1)
            correct_train += (preds == labels).sum().item()
            total_train += len(labels)

        train_loss /= total_train
        train_acc = (correct_train / total_train) * 100

        # Evaluation Loop
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for feats, masks, labels in test_loader:
                feats = feats.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                logits = model(inputs_embeds=feats, attention_mask=masks)
                loss = criterion(logits, labels)

                test_loss += loss.item() * len(labels)
                _, preds = torch.max(logits, dim=-1)
                correct_test += (preds == labels).sum().item()
                total_test += len(labels)

        test_loss /= total_test
        test_acc = (correct_test / total_test) * 100

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.1f}% | "
              f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model checkpoint
            torch.save(model.state_dict(), args.output_model)

        # Save snapshot for resuming
        snapshot_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "num_classes": args.num_classes,
        }
        torch.save(snapshot_data, str(snapshot_path))

    # Training completed successfully, delete temporary snapshot file
    if snapshot_path.exists():
        snapshot_path.unlink()

    sep("Results & Completion")
    print(f"🎉 Fine-Tuning Completed Successfully!")
    print(f"🌟 Best Test Accuracy achieved: {best_acc:.1f}%")
    print(f"💾 Best checkpoint saved to: {args.output_model}")
    print("\nℹ️ To use this fine-tuned checkpoint in the api_server.py or inference.py:")
    print(f"   Simply modify the script to load '{args.output_model}' using standard PyTorch loading:")
    print(f"   model.load_state_dict(torch.load('{args.output_model}'))")
    sep()


if __name__ == "__main__":
    main()
