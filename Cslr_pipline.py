"""
Arabic Continuous Sign Language Recognition (CSLR) - Lightweight Pipeline
==========================================================================

Designed specifically for an EXTREMELY SMALL dataset (only 1-2 video
samples per class). Deep sequential models (LSTM/Transformer) will
overfit badly on this little data, so this pipeline instead:

  1. EXTRACT   - Runs MediaPipe Holistic on every frame of every training
                 video and caches a (T, 96) landmark tensor per video as
                 a .pt file in `data_extracted/`.
                 96 = 6 pose joints * 2 (x,y) + 21 left-hand joints * 2
                    + 21 right-hand joints * 2

  2. TRAIN     - Collapses each variable-length (T, 96) tensor into a
                 single fixed-size 384-dim vector via global statistical
                 pooling (mean, std, max, min across the time axis), then
                 fits a lightweight classifier (RandomForest or SVM) on
                 these pooled vectors. This sidesteps sequence-length
                 mismatches and the overfitting risk of deep models.

  3. INFER     - For a brand-new, continuous video containing several
                 signs in a row: extracts per-frame landmarks, slides a
                 fixed-size window across time, pools + classifies each
                 window, smooths the per-window predictions, merges
                 consecutive duplicate predictions, and renders the final
                 Arabic sentence using `labels_all.csv`.

Expected directory layout
--------------------------
project/
|-- data/
|   |-- 0001/
|   |   |-- 1/              <- frames for instance 1 of class 0001
|   |   |   |-- 0001.jpg
|   |   |   |-- 0002.jpg
|   |   |   `-- ...
|   |   `-- 2/
|   |-- 0080/
|   |   `-- 1/
|   `-- ...
|-- labels_all.csv            (columns: id,gloss,text)
|-- data_extracted/            <- created automatically, caches .pt features
|-- models/                    <- created automatically, stores trained model
`-- cslr_pipeline.py            <- this file

Install dependencies
---------------------
    pip install mediapipe opencv-python torch scikit-learn pandas joblib

Usage
-----
    # Step 1 - one-time feature extraction (safe to re-run; existing
    # .pt files are skipped unless --overwrite is passed)
    python cslr_pipeline.py extract

    # Step 2 - train the lightweight classifier
    python cslr_pipeline.py train --model rf

    # Step 3 - recognize a continuous sentence in a new video
    python cslr_pipeline.py infer --video path/to/sentence.mp4
"""

import os
import glob
import argparse
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import joblib

import mediapipe as mp

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = "data"
EXTRACTED_DIR = "data_extracted"
LABELS_CSV = "labels_all.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "cslr_model.joblib")

# 6 upper-body pose joints (MediaPipe Pose landmark indices).
# Adjust this set if a different set of joints was used during your own
# earlier extraction - the rest of the pipeline doesn't care, as long as
# N_POSE stays consistent with however the cached .pt files were built.
POSE_LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "RIGHT_WRIST": 16,
}
N_POSE = len(POSE_LANDMARKS)          # 6
N_HAND = 21                           # MediaPipe hand landmarks
FEATURES_PER_FRAME = N_POSE * 2 + N_HAND * 2 + N_HAND * 2   # 96

mp_holistic = mp.solutions.holistic


# --------------------------------------------------------------------------
# 1. Feature extraction
# --------------------------------------------------------------------------
class HolisticFeatureExtractor:
    """Wraps mediapipe.solutions.holistic and turns one frame into a
    flat (96,) vector of (x, y) coordinates. Missing landmark groups
    (e.g. a hand that's out of frame) are filled with zeros rather than
    raising an error, so a single bad frame never crashes a whole video.
    """

    def __init__(self, static_image_mode=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self._holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def process_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        pose_feats = np.zeros(N_POSE * 2, dtype=np.float32)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for i, idx in enumerate(POSE_LANDMARKS.values()):
                pose_feats[i * 2] = lm[idx].x
                pose_feats[i * 2 + 1] = lm[idx].y

        left_feats = np.zeros(N_HAND * 2, dtype=np.float32)
        if results.left_hand_landmarks:
            for i, p in enumerate(results.left_hand_landmarks.landmark):
                left_feats[i * 2] = p.x
                left_feats[i * 2 + 1] = p.y

        right_feats = np.zeros(N_HAND * 2, dtype=np.float32)
        if results.right_hand_landmarks:
            for i, p in enumerate(results.right_hand_landmarks.landmark):
                right_feats[i * 2] = p.x
                right_feats[i * 2 + 1] = p.y

        return np.concatenate([pose_feats, left_feats, right_feats]).astype(np.float32)

    def process_frame_dir(self, frame_dir):
        """All .jpg frames in a directory -> (T, 96) array."""
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        feats = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            feats.append(self.process_frame(img))
        if not feats:
            return np.zeros((0, FEATURES_PER_FRAME), dtype=np.float32)
        return np.stack(feats)

    def process_video(self, video_path, frame_stride=1):
        """A continuous video file or folder of frames -> (T, 96) array."""
        if os.path.isdir(video_path):
            return self.process_frame_dir(video_path)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        feats = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                feats.append(self.process_frame(frame))
            idx += 1
        cap.release()
        if not feats:
            return np.zeros((0, FEATURES_PER_FRAME), dtype=np.float32)
        return np.stack(feats)


def extract_dataset(data_dir=DATA_DIR, out_dir=EXTRACTED_DIR, overwrite=False):
    """Walk data/<class_id>/<instance>/*.jpg, run MediaPipe on every
    instance folder, and cache the result as data_extracted/<class_id>_<instance>.pt
    """
    os.makedirs(out_dir, exist_ok=True)
    class_dirs = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
    print(f"Found {len(class_dirs)} class folders in '{data_dir}'.")

    with HolisticFeatureExtractor() as extractor:
        for class_id in class_dirs:
            class_path = os.path.join(data_dir, class_id)
            instance_dirs = sorted(d for d in os.listdir(class_path)
                                    if os.path.isdir(os.path.join(class_path, d)))
            for inst in instance_dirs:
                out_name = f"{class_id}_{inst}.pt"
                out_path = os.path.join(out_dir, out_name)
                if os.path.exists(out_path) and not overwrite:
                    continue
                frame_dir = os.path.join(class_path, inst)
                feats = extractor.process_frame_dir(frame_dir)
                if feats.shape[0] == 0:
                    print(f"  [WARN] no readable frames in {frame_dir}, skipping.")
                    continue
                torch.save(torch.from_numpy(feats), out_path)
                print(f"  saved {out_path}  shape={tuple(feats.shape)}")
    print("Feature extraction complete.")


# --------------------------------------------------------------------------
# 2. Global statistical pooling + training
# --------------------------------------------------------------------------
def pooled_stats(feat_tensor):
    """(T, 96) -> (384,) vector: [mean | std | max | min] across time.
    This is what lets a classical ML model (RF/SVM) consume sequences of
    any length without ever seeing the raw time axis - it just sees a
    fixed-size statistical summary of the motion, which is far less prone
    to overfitting than a sequence model would be on 1-2 samples/class.
    """
    arr = feat_tensor.numpy() if isinstance(feat_tensor, torch.Tensor) else feat_tensor
    if arr.shape[0] == 0:
        return np.zeros(FEATURES_PER_FRAME * 4, dtype=np.float32)
    if arr.shape[0] == 1:
        mean = std = mx = mn = arr[0]
        std = np.zeros_like(mean)
    else:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        mx = arr.max(axis=0)
        mn = arr.min(axis=0)
    return np.concatenate([mean, std, mx, mn]).astype(np.float32)


def load_labels(csv_path=LABELS_CSV):
    """labels_all.csv (id,gloss,text) -> {id: {"gloss":..., "text":...}}"""
    df = pd.read_csv(csv_path, dtype={"id": str})
    return {row["id"]: {"gloss": row["gloss"], "text": row["text"]}
            for _, row in df.iterrows()}


def build_training_set(extracted_dir=EXTRACTED_DIR, labels_csv=LABELS_CSV):
    label_map = load_labels(labels_csv)
    pt_files = sorted(glob.glob(os.path.join(extracted_dir, "*.pt")))
    if not pt_files:
        raise RuntimeError(
            f"No .pt feature files found in '{extracted_dir}'. Run "
            f"`python cslr_pipeline.py extract` first."
        )

    X, y_ids, sample_names = [], [], []
    for pt_path in pt_files:
        fname = os.path.basename(pt_path)            # e.g. "0080_1.pt"
        class_id = fname.split("_")[0]
        if class_id not in label_map:
            print(f"  [WARN] class id '{class_id}' (from {fname}) not in "
                  f"{labels_csv}, skipping.")
            continue
        feats = torch.load(pt_path)
        X.append(pooled_stats(feats))
        y_ids.append(class_id)
        sample_names.append(fname)

    return np.stack(X), np.array(y_ids), sample_names, label_map


def train_model(model_type="rf", extracted_dir=EXTRACTED_DIR,
                 labels_csv=LABELS_CSV, out_path=MODEL_FILE):
    X, y_ids, sample_names, label_map = build_training_set(extracted_dir, labels_csv)
    n_classes = len(set(y_ids))
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} pooled features, {n_classes} classes.")

    counts = Counter(y_ids)
    n_single = sum(1 for c in counts.values() if c == 1)
    if n_single:
        print(f"  [NOTE] {n_single} of {n_classes} classes have only ONE sample. "
              f"Leave-one-out accuracy for those classes is not meaningful "
              f"(the model never sees a same-class example during validation) - "
              f"treat the reported score below as an optimistic upper bound, "
              f"not a true generalization estimate.")

    le = LabelEncoder()
    y = le.fit_transform(y_ids)

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif model_type == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                  class_weight="balanced", random_state=42)
    else:
        raise ValueError("model_type must be 'rf' or 'svm'")

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    try:
        y_pred = cross_val_predict(pipeline, X, y, cv=LeaveOneOut(), n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        print(f"Leave-one-out accuracy (optimistic estimate): {acc:.3f}")
    except Exception as e:
        print(f"  [WARN] could not run leave-one-out evaluation: {e}")

    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipeline, "label_encoder": le, "label_map": label_map}, out_path)
    print(f"Model saved to '{out_path}'.")
    return pipeline, le, label_map


# --------------------------------------------------------------------------
# 3. Sliding-window inference on a continuous video
# --------------------------------------------------------------------------
def _build_windows(n_frames, window_size, stride):
    if n_frames <= window_size:
        return [(0, n_frames)]
    windows, start = [], 0
    while start + window_size <= n_frames:
        windows.append((start, start + window_size))
        start += stride
    if windows[-1][1] < n_frames:
        windows.append((n_frames - window_size, n_frames))
    return windows


def sliding_window_predict(video_path, model_path=MODEL_FILE, window_size=30,
                            stride=15, conf_threshold=0.55, smoothing=3,
                            frame_stride=1, verbose=True):
    """Recognize a continuous video containing several signs in a row.

    Steps: extract per-frame landmarks for the whole video -> slide a
    fixed-size window across time -> pool + classify each window ->
    reject low-confidence windows -> smooth with a small majority-vote
    filter -> collapse consecutive repeats -> map class ids to Arabic
    text via labels_all.csv.

    Returns (merged_class_ids, merged_words, sentence_string).
    """
    bundle = joblib.load(model_path)
    pipeline, le, label_map = bundle["pipeline"], bundle["label_encoder"], bundle["label_map"]

    with HolisticFeatureExtractor() as extractor:
        feats = extractor.process_video(video_path, frame_stride=frame_stride)

    n_frames = feats.shape[0]
    if n_frames == 0:
        print("No frames / landmarks detected in the video.")
        return [], [], ""

    windows = _build_windows(n_frames, window_size, stride)

    raw_preds = []  # list of (class_id_or_None, confidence)
    for (s, e) in windows:
        pooled = pooled_stats(feats[s:e]).reshape(1, -1)
        probs = pipeline.predict_proba(pooled)[0]
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])
        class_id = le.inverse_transform([top_idx])[0]
        accepted = conf >= conf_threshold
        raw_preds.append((class_id if accepted else None, conf))
        if verbose:
            gloss = label_map.get(class_id, {}).get("gloss", "?")
            tag = "" if accepted else "  [below threshold, ignored]"
            print(f"  window[{s:>4}:{e:>4}] -> {class_id} ({gloss})  conf={conf:.2f}{tag}")

    # Majority-vote smoothing over a small neighborhood to remove flicker
    half = max(smoothing // 2, 0)
    smoothed = []
    for i in range(len(raw_preds)):
        lo, hi = max(0, i - half), min(len(raw_preds), i + half + 1)
        neighborhood = [p for p, _ in raw_preds[lo:hi] if p is not None]
        smoothed.append(Counter(neighborhood).most_common(1)[0][0] if neighborhood else None)

    # Collapse consecutive identical predictions into one event
    merged_ids = []
    for cid in smoothed:
        if cid is None:
            continue
        if not merged_ids or merged_ids[-1] != cid:
            merged_ids.append(cid)

    merged_words = [label_map.get(cid, {}).get("text", f"<{cid}>") for cid in merged_ids]
    sentence = " ".join(merged_words)

    if verbose:
        print(f"\nDetected class id sequence: {merged_ids}")
        print(f"Final Arabic sentence: {sentence}")

    return merged_ids, merged_words, sentence


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Arabic CSLR lightweight pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract MediaPipe features from data/ into data_extracted/")
    p_extract.add_argument("--data_dir", default=DATA_DIR)
    p_extract.add_argument("--out_dir", default=EXTRACTED_DIR)
    p_extract.add_argument("--overwrite", action="store_true")

    p_train = sub.add_parser("train", help="Train the lightweight classifier on cached features")
    p_train.add_argument("--model", choices=["rf", "svm"], default="rf")
    p_train.add_argument("--extracted_dir", default=EXTRACTED_DIR)
    p_train.add_argument("--labels_csv", default=LABELS_CSV)
    p_train.add_argument("--out", default=MODEL_FILE)

    p_infer = sub.add_parser("infer", help="Run sliding-window recognition on a continuous video")
    p_infer.add_argument("--video", required=True)
    p_infer.add_argument("--model_path", default=MODEL_FILE)
    p_infer.add_argument("--window", type=int, default=30, help="window size in frames")
    p_infer.add_argument("--stride", type=int, default=15, help="step between windows in frames")
    p_infer.add_argument("--threshold", type=float, default=0.55, help="min confidence to accept a window prediction")
    p_infer.add_argument("--smoothing", type=int, default=3, help="majority-vote neighborhood size (odd number)")
    p_infer.add_argument("--frame_stride", type=int, default=1, help="process every Nth frame (speed/accuracy tradeoff)")

    args = parser.parse_args()

    if args.command == "extract":
        extract_dataset(args.data_dir, args.out_dir, args.overwrite)
    elif args.command == "train":
        train_model(args.model, args.extracted_dir, args.labels_csv, args.out)
    elif args.command == "infer":
        sliding_window_predict(
            video_path=args.video, model_path=args.model_path,
            window_size=args.window, stride=args.stride,
            conf_threshold=args.threshold, smoothing=args.smoothing,
            frame_stride=args.frame_stride,
        )


if __name__ == "__main__":
    main()