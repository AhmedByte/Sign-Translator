import argparse
import csv
import json
import os
from pathlib import Path
import re

import torch
import torch.nn as nn

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
        for idx, row in enumerate(reader):
            labels[idx] = {
                "id": row["id"].strip(),
                "gloss": row.get("gloss", "").strip(),
                "text": row.get("text", "").strip(),
            }
    return labels

# ───────────────────────────────────────────────────────────────────────────
# Sliding Window Inference
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model_lightweight.pt")
    parser.add_argument("--labels_name", default="labels_all.csv")
    parser.add_argument("--input_video_frames", required=True, help="Directory containing frames of the continuous video")
    parser.add_argument("--window_size", type=int, default=25, help="Number of frames per window")
    parser.add_argument("--stride", type=int, default=5, help="Step size between windows")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold to accept a sign")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Labels
    labels = load_labels_csv(args.labels_name)
    num_classes = len(labels)
    if num_classes == 0:
        print("❌ No labels found.")
        return

    # 2. Load Model
    if not os.path.exists(args.model_path):
        print(f"❌ Model {args.model_path} not found. Please train it first using train_lightweight.py")
        return
        
    model = SignLSTM(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 3. Extract Features for the whole continuous video
    print(f"⏳ Extracting features from continuous video frames: {args.input_video_frames} ...")
    feats = extract_landmarks_from_frames(Path(args.input_video_frames))
    if len(feats) == 0:
        print("❌ Failed to extract features or empty folder.")
        return
        
    print(f"✅ Extracted {len(feats)} frames.")

    # 4. Sliding Window over the sequence
    predictions = []
    T = len(feats)
    
    if T < args.window_size:
        print("⚠️ Video is shorter than the window size. Running single prediction.")
        windows = [(0, T)]
    else:
        windows = [(i, i + args.window_size) for i in range(0, T - args.window_size + 1, args.stride)]

    print(f"🔍 Running sliding window (Window: {args.window_size}, Stride: {args.stride})...")
    
    with torch.no_grad():
        for start, end in windows:
            window_feats = feats[start:end].unsqueeze(0).to(device) # Shape: (1, window_size, 96)
            seq_lengths = torch.tensor([end - start])
            
            logits = model(window_feats, seq_lengths)
            probs = torch.softmax(logits, dim=-1)[0]
            
            conf, pred_id = torch.max(probs, dim=-1)
            conf = conf.item()
            pred_id = pred_id.item()
            
            if conf > args.threshold:
                predictions.append({
                    "start": start,
                    "end": end,
                    "pred_id": pred_id,
                    "confidence": conf,
                    "text": labels.get(pred_id, {}).get("text", str(pred_id)),
                    "gloss": labels.get(pred_id, {}).get("gloss", "")
                })

    # 5. Merge consecutive identical predictions (like CTC decoding)
    if not predictions:
        print("🤷‍♂️ No signs detected above the confidence threshold.")
        return

    final_sequence = []
    current_sign = predictions[0]

    for p in predictions[1:]:
        if p["pred_id"] == current_sign["pred_id"]:
            # Extend current sign duration
            current_sign["end"] = p["end"]
            current_sign["confidence"] = max(current_sign["confidence"], p["confidence"])
        else:
            final_sequence.append(current_sign)
            current_sign = p
    final_sequence.append(current_sign)

    print("\n" + "="*50)
    print("📝 Detected Sentence / Sequence:")
    print("="*50)
    sentence = " ".join([sign["text"] for sign in final_sequence])
    print(f"🌟 {sentence}")
    print("="*50)
    print("Detailed Timeline:")
    for sign in final_sequence:
        print(f"  [{sign['start']:>3d} -> {sign['end']:>3d}] ➡️ {sign['text']:<15} (Conf: {sign['confidence']*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
