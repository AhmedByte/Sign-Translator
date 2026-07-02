"""
utils/data_utils.py

Utility functions for:
  • Loading and parsing config.json (id2label, model hyper-parameters).
  • Mapping predicted token IDs back to human-readable sign labels.
  • Generating mock/dummy video feature tensors for smoke-testing inference.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Config helpers                                                               #
# --------------------------------------------------------------------------- #

def load_config(config_path: str) -> dict:
    """
    Load and return the raw config.json as a Python dictionary.

    Parameters
    ----------
    config_path : str
        Path to ``config.json`` (the HuggingFace model config).

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    return config


def build_id2label(config: dict) -> Dict[int, str]:
    """
    Extract the ``id2label`` mapping from a parsed config dict.

    The keys in the JSON may be stored as strings (JSON limitation); this
    function coerces them to integers so look-ups work with raw token IDs.

    Parameters
    ----------
    config : dict
        Parsed config.json content (output of :func:`load_config`).

    Returns
    -------
    Dict[int, str]
        Mapping from integer token ID → sign label string.

    Example
    -------
    >>> cfg  = load_config("config.json")
    >>> id2l = build_id2label(cfg)
    >>> id2l[0]
    'ا'
    """
    raw: dict = config.get("id2label", {})
    if not raw:
        raise ValueError(
            "config.json does not contain an 'id2label' field.  "
            "Please verify the checkpoint."
        )
    return {int(k): v for k, v in raw.items()}


def build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
    """
    Invert an id→label mapping to label→id.

    Parameters
    ----------
    id2label : Dict[int, str]

    Returns
    -------
    Dict[str, int]
    """
    return {v: k for k, v in id2label.items()}


# --------------------------------------------------------------------------- #
# Decoding helpers                                                             #
# --------------------------------------------------------------------------- #

# Special token IDs used by BART (standard values; override if your config
# uses different IDs via config["bos_token_id"] etc.)
_SPECIAL_IDS = {0, 1, 2}  # pad, eos, bos in most BART checkpoints


def decode_token_ids(
    token_ids: List[int],
    id2label: Dict[int, str],
    skip_special_tokens: bool = True,
) -> str:
    """
    Convert a list of decoder output token IDs to a readable label string.

    Parameters
    ----------
    token_ids : List[int]
        Output of ``model.generate()[0].tolist()``.
    id2label : Dict[int, str]
        Mapping from token ID to label (built by :func:`build_id2label`).
    skip_special_tokens : bool
        When ``True``, tokens with IDs 0, 1, or 2 (pad / eos / bos) are
        excluded from the output.

    Returns
    -------
    str
        Space-joined label string, e.g. ``"أنا أحبك"``.
    """
    labels: List[str] = []
    for tid in token_ids:
        if skip_special_tokens and tid in _SPECIAL_IDS:
            continue
        label = id2label.get(tid)
        if label is not None:
            labels.append(label)
        else:
            labels.append(f"<UNK:{tid}>")
    return " ".join(labels)


def decode_batch(
    batch_token_ids: torch.Tensor,
    id2label: Dict[int, str],
    skip_special_tokens: bool = True,
) -> List[str]:
    """
    Decode a batch of generated sequences.

    Parameters
    ----------
    batch_token_ids : torch.Tensor, shape (batch_size, seq_len)
        Tensor returned by ``model.generate()``.
    id2label : Dict[int, str]
    skip_special_tokens : bool

    Returns
    -------
    List[str]
        One decoded string per sample in the batch.
    """
    return [
        decode_token_ids(row.tolist(), id2label, skip_special_tokens)
        for row in batch_token_ids
    ]


# --------------------------------------------------------------------------- #
# Mock / dummy feature generator                                               #
# --------------------------------------------------------------------------- #

def create_mock_video_features(
    batch_size: int = 2,
    sequence_length: int = 30,
    feature_dim: int = 256,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random video feature tensors that mimic real pose/skeleton features
    extracted from sign-language video clips.

    Parameters
    ----------
    batch_size : int
        Number of video clips in the batch.
    sequence_length : int
        Number of frames (time steps) per clip.
    feature_dim : int
        Dimensionality of each frame's feature vector.  Must match
        ``config.input_feature_dim`` (default 256 for this checkpoint).
    seed : int, optional
        Random seed for reproducibility.  ``None`` disables seeding.
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    features : torch.Tensor, shape (batch_size, sequence_length, feature_dim)
        Normalised Gaussian features.
    attention_mask : torch.Tensor, shape (batch_size, sequence_length)
        All-ones mask (every frame is valid).  Replace individual rows with
        zeros past the true sequence length when using variable-length clips.

    Example
    -------
    >>> features, mask = create_mock_video_features(batch_size=1, sequence_length=20)
    >>> features.shape
    torch.Size([1, 20, 256])
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Gaussian features, unit-normalised along the feature dimension to
    # roughly mimic L2-normalised skeleton descriptors.
    features = torch.randn(batch_size, sequence_length, feature_dim, device=device)
    features = torch.nn.functional.normalize(features, dim=-1)

    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)

    return features, attention_mask


# --------------------------------------------------------------------------- #
# Feature file loader (real inference)                                        #
# --------------------------------------------------------------------------- #

def load_features_from_file(
    file_path: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load pre-extracted video features from a ``.pt`` (PyTorch) or ``.npy``
    (NumPy) file and return them as a batched tensor.

    Expected shapes
    ---------------
    * ``.pt``  file: a ``torch.Tensor`` of shape ``(T, D)`` or ``(1, T, D)``.
    * ``.npy`` file: a NumPy array of shape ``(T, D)`` or ``(1, T, D)``.

    Parameters
    ----------
    file_path : str
        Path to the saved feature file.
    device : str

    Returns
    -------
    features : torch.Tensor, shape (1, T, D)
    attention_mask : torch.Tensor, shape (1, T)
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pt":
        features = torch.load(file_path, map_location=device)
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor in {file_path}, got {type(features)}")
    elif suffix == ".npy":
        import numpy as np
        features = torch.from_numpy(np.load(str(file_path))).to(device)
    else:
        raise ValueError(
            f"Unsupported feature file format '{suffix}'.  "
            "Provide a .pt or .npy file."
        )

    # Ensure shape is (1, T, D)
    if features.dim() == 2:
        features = features.unsqueeze(0)
    elif features.dim() != 3:
        raise ValueError(
            f"Feature tensor must be 2-D (T, D) or 3-D (B, T, D), "
            f"got {features.dim()}-D."
        )

    features = features.float().to(device)
    attention_mask = torch.ones(
        features.shape[0], features.shape[1],
        dtype=torch.long, device=device,
    )
    return features, attention_mask
