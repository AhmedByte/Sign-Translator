# SignBart — Sign Language Recognition

A Python project for running inference with the **SignBart** model
(`tinh2312/SignBart-KArSL03-502`), a BART-based sequence-to-sequence
architecture that maps pre-extracted video feature sequences to Arabic sign
language labels (502 classes).

---

## Project Structure

```
sign_language_project/
│
├── models/
│   ├── __init__.py
│   └── sign_bart.py        # Custom SignBart class (encoder accepts video features)
│
├── utils/
│   ├── __init__.py
│   └── data_utils.py       # Config loading, id2label mapping, feature helpers
│
├── inference.py            # Main inference script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 |
| transformers | ≥ 4.40 |
| safetensors | ≥ 0.4.3 |

---

## Installation

```bash
# 1. Clone / download this project
cd sign_language_project

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

For **GPU inference**, install the correct CUDA build of PyTorch first:

```bash
# Example — PyTorch 2.2 + CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Downloading the Checkpoint

Place `config.json` and `model.safetensors` in the same directory
(the project root by default, or any folder you pass to `--checkpoint_dir`).

```bash
# Option A — huggingface-hub CLI
huggingface-cli download tinh2312/SignBart-KArSL03-502 \
    --local-dir ./checkpoint

# Option B — Python
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("tinh2312/SignBart-KArSL03-502", local_dir="./checkpoint")
EOF
```

---

## Running Inference

### Quick smoke test (mock / dummy features)

```bash
python inference.py --checkpoint_dir .
```

Expected output:

```
============================== SignBart Sign Language Recognition — Inference ==============================
  Device            : cpu

[1/4] Loading config …
  d_model           : 256
  input_feature_dim : 256
  vocab_size        : 4
  num_labels        : 502
  ...

[2/4] Building model and loading safetensors weights …
  Total parameters  : X,XXX,XXX

[3/4] Preparing input features …
  No --features path supplied.  Using mock features (batch=2, seq_len=30, dim=256).
  video_features shape  : (2, 30, 256)
  attention_mask shape  : (2, 30)

[4/4] Running model.generate() …
...

================================== Results ==================================

  Sample 1:
    Generated token IDs : [2, 42, 7, ...]
    Predicted labels    : أنا أحبك

  Sample 2:
    ...
```

### Inference on real pre-extracted features

Your features must be saved as a **PyTorch tensor** (`.pt`) or **NumPy array**
(`.npy`) of shape `(T, 256)` or `(1, T, 256)`, where `T` is the number of
video frames.

```bash
python inference.py \
    --checkpoint_dir ./checkpoint \
    --features       path/to/my_features.pt \
    --max_new_tokens 30 \
    --beam_size      4
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir` | `.` | Folder with `config.json` + `model.safetensors` |
| `--features` | `None` | Path to `.pt` / `.npy` feature file |
| `--input_feature_dim` | `d_model` (256) | Raw feature dimensionality |
| `--batch_size` | `2` | Batch size for mock features |
| `--seq_len` | `30` | Frame count for mock features |
| `--max_new_tokens` | `20` | Max tokens per output sequence |
| `--beam_size` | `4` | Beam width for beam-search |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

---

## Architecture Notes

### Encoder

The BART encoder receives **pre-extracted video features** rather than token
embeddings.  A `feature_projection` linear layer (`input_feature_dim →
d_model`) is applied first, then the projected vectors are fed directly to the
`BartEncoder` as `inputs_embeds`.

### Decoder

The BART decoder auto-regressively generates token IDs that are mapped back to
sign labels via the `id2label` table from `config.json` (502 Arabic sign
classes).

### Weight loading

Weights are loaded with `safetensors.torch.load_file` and applied via
`model.load_state_dict(..., strict=False)`.  The `feature_projection` layer is
the only key that may appear as *missing* when the checkpoint was saved without
it; all other encoder / decoder / classification-head weights should align
exactly.

---

## Extending to Real Video

For production use, replace the mock features with real ones extracted from
video frames.  A typical pipeline:

```
Video frames
    │
    ▼
Pose / landmark extraction   (e.g. MediaPipe, OpenPose)
    │
    ▼
Feature encoding             (e.g. MLP, GCN, or raw keypoint flattening → dim 256)
    │
    ▼
Temporal stack               (T × 256 tensor)
    │
    ▼
inference.py  --features my_clip.pt
```

---

## License

This project is released for research and educational purposes.
Model weights belong to the original authors (`tinh2312/SignBart-KArSL03-502`).
