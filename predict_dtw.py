import argparse
import csv
import json
import os
from pathlib import Path
import re
import math

import torch
import torch.nn as nn

# ───────────────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────────────
def extract_landmarks_from_frames(frames_dir: Path) -> torch.Tensor:
    import cv2
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except ImportError:
        raise ImportError("Please install mediapipe: pip install mediapipe")

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]
    
    img_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=natural_sort_key)
    if not img_paths:
        return torch.tensor([])

    features = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for img_path in img_paths:
            frame = cv2.imread(str(img_path))
            if frame is None: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            pose_x, pose_y = [0.0]*6, [0.0]*6
            if results.pose_landmarks:
                for j_idx, joint in enumerate([11, 12, 13, 14, 15, 16]):
                    lm = results.pose_landmarks.landmark[joint]
                    pose_x[j_idx], pose_y[j_idx] = lm.x, lm.y

            lh_x, lh_y = [0.0]*21, [0.0]*21
            if results.left_hand_landmarks:
                for h_idx in range(21):
                    lm = results.left_hand_landmarks.landmark[h_idx]
                    lh_x[h_idx], lh_y[h_idx] = lm.x, lm.y

            rh_x, rh_y = [0.0]*21, [0.0]*21
            if results.right_hand_landmarks:
                for h_idx in range(21):
                    lm = results.right_hand_landmarks.landmark[h_idx]
                    rh_x[h_idx], rh_y[h_idx] = lm.x, lm.y

            features.append(pose_x + lh_x + rh_x + pose_y + lh_y + rh_y)

    return torch.tensor(features, dtype=torch.float32)

def load_labels_csv(csv_path: str) -> dict:
    labels = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"].strip()] = {
                "gloss": row.get("gloss", "").strip(),
                "text": row.get("text", "").strip(),
            }
    return labels

# ───────────────────────────────────────────────────────────────────────────
# Dynamic Time Warping (DTW) Distance
# ───────────────────────────────────────────────────────────────────────────
def dtw_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """Calculate DTW distance between two sequences of features using Euclidean distance."""
    n, m = len(seq1), len(seq2)
    dtw_matrix = torch.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.norm(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # insertion
                dtw_matrix[i, j - 1],    # deletion
                dtw_matrix[i - 1, j - 1] # match
            )
            
    return dtw_matrix[n, m].item() / (n + m) # Normalize by path length

# ───────────────────────────────────────────────────────────────────────────
# Zero-Shot DTW Continuous Inference
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_extracted", help="Path to cached reference templates")
    parser.add_argument("--labels_name", default="labels_all.csv")
    parser.add_argument("--input_video_frames", required=True, help="Directory containing frames of the continuous video")
    parser.add_argument("--window_size", type=int, default=25, help="Number of frames per window")
    parser.add_argument("--stride", type=int, default=10, help="Step size between windows")
    parser.add_argument("--distance_threshold", type=float, default=0.15, help="Maximum allowed DTW distance to accept a match")
    args = parser.parse_args()

    # 1. Load Labels
    labels = load_labels_csv(args.labels_name)
    if not labels:
        print("❌ No labels found.")
        return

    # 2. Load Templates from Cache
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"❌ Cache directory {args.cache_dir} not found. Ensure features are extracted.")
        return

    templates = {} # Dict of class_id -> list of feature tensors
    for f in cache_path.glob("*.pt"):
        class_id = f.stem.split("_")[0]
        if class_id in labels:
            if class_id not in templates:
                templates[class_id] = []
            feats = torch.load(f, map_location="cpu")
            templates[class_id].append(feats)

    if not templates:
        print("❌ No reference templates found in cache that match the labels.")
        return
        
    print(f"📂 Loaded templates for {len(templates)} classes.")

    # 3. Extract Features from Test Video
    print(f"⏳ Extracting features from continuous video frames: {args.input_video_frames} ...")
    test_feats = extract_landmarks_from_frames(Path(args.input_video_frames))
    if len(test_feats) == 0:
        print("❌ Failed to extract features or empty folder.")
        return
        
    print(f"✅ Extracted {len(test_feats)} frames.")

    # 4. Sliding Window DTW Prediction
    predictions = []
    T = len(test_feats)
    
    if T < args.window_size:
        print("⚠️ Video is shorter than the window size. Running single prediction.")
        windows = [(0, T)]
    else:
        windows = [(i, i + args.window_size) for i in range(0, T - args.window_size + 1, args.stride)]

    print(f"🔍 Running Sliding Window DTW (Window: {args.window_size}, Stride: {args.stride})...")
    
    for start, end in windows:
        window_feats = test_feats[start:end]
        
        best_class = None
        best_dist = float('inf')
        
        # Compare window to all templates
        for class_id, class_templates in templates.items():
            for tmpl in class_templates:
                dist = dtw_distance(window_feats, tmpl)
                if dist < best_dist:
                    best_dist = dist
                    best_class = class_id
                    
        if best_dist < args.distance_threshold and best_class is not None:
            predictions.append({
                "start": start,
                "end": end,
                "class_id": best_class,
                "distance": best_dist,
                "text": labels[best_class].get("text", best_class)
            })

    # 5. Merge consecutive identical predictions
    if not predictions:
        print("🤷‍♂️ No signs detected below the distance threshold.")
        return

    final_sequence = []
    current_sign = predictions[0]

    for p in predictions[1:]:
        if p["class_id"] == current_sign["class_id"]:
            current_sign["end"] = p["end"]
            current_sign["distance"] = min(current_sign["distance"], p["distance"])
        else:
            final_sequence.append(current_sign)
            current_sign = p
    final_sequence.append(current_sign)

    print("\n" + "="*50)
    print("📝 Detected Sentence / Sequence (DTW Zero-shot):")
    print("="*50)
    sentence = " ".join([sign["text"] for sign in final_sequence])
    print(f"🌟 {sentence}")
    print("="*50)
    print("Detailed Timeline:")
    for sign in final_sequence:
        print(f"  [{sign['start']:>3d} -> {sign['end']:>3d}] ➡️ {sign['text']:<15} (Dist: {sign['distance']:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()
