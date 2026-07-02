import argparse
import csv
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def load_labels_csv(csv_path: str) -> set:
    valid_class_ids = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_class_ids.add(row["id"].strip())
    return valid_class_ids

def extract_global_features(feats: torch.Tensor) -> np.ndarray:
    """
    Takes shape (T, 96) and converts to a fixed size 1D vector (384 dimensions)
    by calculating Mean, Std, Max, Min across the time axis (T).
    """
    if len(feats) == 0:
        return np.zeros(96 * 4)
    
    feats_np = feats.numpy()
    mean_f = np.mean(feats_np, axis=0)
    std_f = np.std(feats_np, axis=0)
    max_f = np.max(feats_np, axis=0)
    min_f = np.min(feats_np, axis=0)
    
    return np.concatenate([mean_f, std_f, max_f, min_f])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_extracted", help="Path to cached feature files")
    parser.add_argument("--labels_name", default="labels_all.csv", help="Labels CSV file")
    parser.add_argument("--output_model", default="model_rf.joblib")
    args = parser.parse_args()

    # 1. Read labels
    labels_file = Path(args.labels_name)
    if not labels_file.exists():
        print(f"❌ Labels file {args.labels_name} not found.")
        return
    valid_class_ids = load_labels_csv(str(labels_file))
    
    # 2. Load Cached Data
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache directory {args.cache_dir} not found.")
        return

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

    sorted_classes = sorted(list(class_to_samples.keys()))
    num_classes = len(sorted_classes)
    print(f"📂 Found {num_classes} classes.")

    X_train, y_train = [], []
    X_test, y_test = [], []

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
            pooled = extract_global_features(feats)
            X_train.append(pooled)
            y_train.append(label_idx)
            
        for tf in test_files:
            feats = torch.load(tf, map_location="cpu")
            pooled = extract_global_features(feats)
            X_test.append(pooled)
            y_test.append(label_idx)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"📊 Train Samples : {len(X_train)}")
    print(f"📊 Test Samples  : {len(X_test)}")
    print(f"⚙️ Feature Vector Size: {X_train.shape[1]}")

    # 3. Train Model (Random Forest or SVM)
    print("\n⏳ Training Random Forest Classifier (Best for small data) ...")
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds) * 100
    test_acc = accuracy_score(y_test, test_preds) * 100

    print(f"✅ Train Acc: {train_acc:.1f}%")
    print(f"✅ Test Acc: {test_acc:.1f}%")

    # Save model
    joblib.dump(model, args.output_model)
    print(f"💾 Model saved to: {args.output_model}")

if __name__ == "__main__":
    main()
