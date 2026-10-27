"""Small task head used by the optional offline evaluation extra."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from safetensors.torch import load_file, save_file


class GradedEncoder(nn.Module):
    def __init__(self, backbone, objective="ce"):
        super().__init__()
        self.backbone = backbone
        self.objective = objective
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(
            backbone.config.hidden_size, 4 if objective == "corn" else 5
        )

    def forward(self, **inputs):
        return self.head(self.dropout(self.backbone(**inputs).last_hidden_state[:, 0]))

    def probabilities(self, logits):
        if self.objective != "corn":
            return logits.float().softmax(-1)
        survival = logits.float().sigmoid().cumprod(-1)
        return torch.cat(
            [1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]],
            -1,
        )

    def loss(self, logits, targets):
        if self.objective == "corn":
            # P(Y > k | Y > k-1), k=0..3 for zero-based labels.
            cuts = torch.arange(4, device=targets.device)
            eligible = targets[:, None] > cuts - 1
            levels = (targets[:, None] > cuts).float()
            return F.binary_cross_entropy_with_logits(
                logits.float(), levels, reduction="none"
            )[eligible].mean()
        probs = self.probabilities(logits)
        scores = probs @ torch.arange(1, 6, device=logits.device, dtype=torch.float32)
        if self.objective == "mse":
            return F.mse_loss(scores, targets.float() + 1)
        loss = F.cross_entropy(logits.float(), targets)
        if self.objective == "ce_rank":
            direction = (targets[:, None] - targets[None, :]).sign()
            mask = direction != 0
            if mask.any():
                loss = (
                    loss
                    + 0.3
                    * F.softplus(-(scores[:, None] - scores[None, :]) * direction)[
                        mask
                    ].mean()
                )
        return loss

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "encoder", safe_serialization=True)
        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.head.state_dict().items()
            },
            str(path / "head.safetensors"),
        )
        (path / "head_config.json").write_text(
            json.dumps({"objective": self.objective}) + "\n"
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        config = json.loads((path / "head_config.json").read_text())
        model = cls(
            AutoModel.from_pretrained(path / "encoder", local_files_only=True),
            config["objective"],
        )
        model.head.load_state_dict(load_file(str(path / "head.safetensors")))
        return model
