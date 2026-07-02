"""
models/sign_bart.py  (v4 – matches checkpoint projection & classification head)
"""

import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder


class SignBartConfig(BartConfig):
    model_type = "SignBart"

    def __init__(self, coord_dim: int = 48, num_labels: int = 502, **kwargs):
        super().__init__(**kwargs)
        self.coord_dim = coord_dim
        self.num_labels = num_labels


class SignBartProjection(nn.Module):
    """Projects (x, y) landmark coordinates into d_model space."""
    def __init__(self, coord_dim: int, d_model: int):
        super().__init__()
        self.proj_x1 = nn.Linear(coord_dim, d_model)
        self.proj_y1 = nn.Linear(coord_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.proj_x1(x) + self.proj_y1(y)
        return self.dropout(out)


class ClassificationHead(nn.Module):
    """Single linear projection → num_labels."""
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.dropout(x))


class SignBart(nn.Module):
    def __init__(self, config: SignBartConfig):
        super().__init__()
        self.config = config

        # 1. Project (x, y) landmark coords → d_model
        self.projection = SignBartProjection(config.coord_dim, config.d_model)

        # 2. Encoder & Decoder
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

        # 3. Classification head → num_labels classes
        self.classification_head = ClassificationHead(config.d_model, config.num_labels)

    def forward(
        self,
        inputs_embeds: torch.Tensor,            # (B, T, coord_dim * 2)
        attention_mask: torch.Tensor = None,     # (B, T)
        decoder_input_ids: torch.Tensor = None,
    ):
        B, T, _ = inputs_embeds.shape
        device = inputs_embeds.device
        cd = self.config.coord_dim

        # 1. Split into x / y coordinates and project
        x_coords = inputs_embeds[:, :, :cd]      # (B, T, coord_dim)
        y_coords = inputs_embeds[:, :, cd:]       # (B, T, coord_dim)
        projected = self.projection(x_coords, y_coords)   # (B, T, d_model)

        # 2. Encode
        enc_out = self.encoder(
            input_ids=None,
            inputs_embeds=projected,
            attention_mask=attention_mask,
        )
        enc_hidden = enc_out.last_hidden_state   # (B, T, d_model)

        # 3. Decode with a single BOS token
        if decoder_input_ids is None:
            bos = self.config.decoder_start_token_id or self.config.bos_token_id or 0
            decoder_input_ids = torch.full((B, 1), bos, dtype=torch.long, device=device)

        dec_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=attention_mask,
        )
        dec_hidden = dec_out.last_hidden_state   # (B, 1, d_model)

        # 4. Classify using BOS hidden state
        pooled = dec_hidden[:, 0, :]             # (B, d_model)
        logits = self.classification_head(pooled)  # (B, num_labels)
        return logits

    @classmethod
    def from_safetensors(cls, config, safetensors_path, device="cpu"):
        from safetensors.torch import load_file

        model = cls(config)
        ckpt  = load_file(safetensors_path, device=device)

        # Handle classification head resizing if size changed (transfer learning)
        ckpt_head_weight = ckpt.get("classification_head.out_proj.weight")
        ckpt_head_bias = ckpt.get("classification_head.out_proj.bias")
        
        if ckpt_head_weight is not None and ckpt_head_weight.shape != model.classification_head.out_proj.weight.shape:
            print(f"[SignBart] Resizing classification head from {ckpt_head_weight.shape[0]} to {config.num_labels} classes.")
            
            # Create new parameter tensors
            new_weight = nn.Parameter(torch.randn(config.num_labels, config.d_model) * 0.02)
            new_bias = nn.Parameter(torch.zeros(config.num_labels))
            
            # Copy old weights for the overlapping classes
            min_classes = min(ckpt_head_weight.shape[0], config.num_labels)
            with torch.no_grad():
                new_weight[:min_classes].copy_(ckpt_head_weight[:min_classes])
                if ckpt_head_bias is not None:
                    new_bias[:min_classes].copy_(ckpt_head_bias[:min_classes])
            
            # Replace the parameters in the model
            model.classification_head.out_proj.weight = new_weight
            model.classification_head.out_proj.bias = new_bias
            
            # Remove from checkpoint dict to prevent strict/load mismatches
            del ckpt["classification_head.out_proj.weight"]
            if "classification_head.out_proj.bias" in ckpt:
                del ckpt["classification_head.out_proj.bias"]

        # Keys now match directly
        missing, unexpected = model.load_state_dict(ckpt, strict=False)

        if missing:
            print(f"[SignBart] Missing ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[SignBart] Unexpected ({len(unexpected)}): {unexpected[:8]}")

        model.to(device).eval()
        print(f"[SignBart] Weights loaded ✓")
        return model