"""
inference.py  (v2 - corrected for SignBart classifier architecture)

Usage:
    python inference.py --checkpoint_dir .
    python inference.py --checkpoint_dir . --features my_clip.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sign_bart import SignBart, SignBartConfig


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels_csv(csv_path: str) -> dict:
    """Load labels.csv → {index: {"id": "0001", "gloss": "...", "text": "..."}}."""
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


def build_config(raw: dict) -> SignBartConfig:
    ALLOWED = {
        "vocab_size", "d_model", "encoder_layers", "decoder_layers",
        "encoder_attention_heads", "decoder_attention_heads",
        "encoder_ffn_dim", "decoder_ffn_dim",
        "max_position_embeddings", "dropout", "attention_dropout",
        "activation_function", "pad_token_id", "bos_token_id",
        "eos_token_id", "decoder_start_token_id", "is_encoder_decoder",
        "scale_embedding", "use_cache",
    }
    kwargs = {k: v for k, v in raw.items() if k in ALLOWED}
    num_labels = len(raw.get("id2label", {}))
    coord_dim = raw.get("coord_dim", 48)
    return SignBartConfig(
        coord_dim=coord_dim,
        num_labels=num_labels,
        **kwargs,
    )


def mock_features(batch=2, seq=30, coord_dim=48, device="cpu"):
    torch.manual_seed(42)
    # simulate (x, y) landmark coordinates: 48 x-coords + 48 y-coords = 96
    feats = torch.randn(batch, seq, coord_dim * 2, device=device)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    return feats, mask


def sep(title="", w=65):
    if title:
        p = (w - len(title) - 2) // 2
        print("=" * p + f" {title} " + "=" * max(0, w - p - len(title) - 2))
    else:
        print("=" * w)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(args):
    sep("SignBart Inference")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device : {device}")

    # 1. Config
    print("\n[1/4] Loading config …")
    raw = load_config(str(Path(args.checkpoint_dir) / "config.json"))
    config = build_config(raw)

    # Load human-readable labels from CSV
    labels_path = Path(args.checkpoint_dir) / "labels.csv"
    if labels_path.exists():
        labels = load_labels_csv(str(labels_path))
        print(f"  Labels     : loaded {len(labels)} from labels.csv ✓")
    else:
        labels = {}
        print(f"  Labels     : labels.csv not found, using raw IDs")

    print(f"  d_model    : {config.d_model}")
    print(f"  coord_dim  : {config.coord_dim}")
    print(f"  num_labels : {config.num_labels}")
    print(f"  vocab_size : {config.vocab_size}")

    # 2. Model
    print("\n[2/4] Loading model weights …")
    weights_path = str(Path(args.checkpoint_dir) / "model.safetensors")
    model = SignBart.from_safetensors(config, weights_path, device=device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}")

    # 3. Features
    print("\n[3/4] Preparing input features …")
    if args.features:
        raw_feats = torch.load(args.features, map_location=device)
        if raw_feats.dim() == 2:
            raw_feats = raw_feats.unsqueeze(0)
        feats = raw_feats.float()
        mask = torch.ones(feats.shape[0], feats.shape[1], dtype=torch.long, device=device)
        print(f"  Loaded from file : {args.features}")
    else:
        feats, mask = mock_features(args.batch_size, args.seq_len, config.coord_dim, device)
        print(f"  Using mock features (batch={args.batch_size}, seq={args.seq_len}, coords={config.coord_dim}×2)")
    print(f"  Shape : {tuple(feats.shape)}")

    # 4. Inference
    print("\n[4/4] Running inference …")
    with torch.no_grad():
        logits = model(inputs_embeds=feats, attention_mask=mask)

    print(f"  Logits shape : {tuple(logits.shape)}  (batch × {config.num_labels} classes)")

    # Decode predictions
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_ids = torch.topk(probs, k=5, dim=-1)

    sep("Results")
    for i in range(feats.shape[0]):
        pred_id = top5_ids[i][0].item()
        info = labels.get(pred_id, {})
        gloss = info.get("gloss", f"<ID:{pred_id}>")
        text  = info.get("text", "")
        print(f"\n  Sample {i+1}:")
        print(f"    Top prediction  : [{pred_id}] \"{gloss}\"  "
              f"(confidence: {top5_probs[i][0].item()*100:.1f}%)")
        if text:
            print(f"    Meaning         : {text}")
        print(f"    Top-5 labels    :")
        for rank, (pid, pp) in enumerate(zip(top5_ids[i].tolist(), top5_probs[i].tolist()), 1):
            row = labels.get(pid, {})
            g = row.get("gloss", f"<ID:{pid}>")
            t = row.get("text", "")
            suffix = f"  → {t}" if t else ""
            print(f"      {rank}. [{pid:>3}] {g:<30} {pp*100:.1f}%{suffix}")

    sep()
    print("Done.\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default=".")
    p.add_argument("--features", default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())