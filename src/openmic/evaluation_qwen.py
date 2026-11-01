"""Optional supervised LoRA five-grade classifier with one shared causal base."""

import numpy as np

PREFIX = "请判断以下中文段子的整体好笑程度，等级为1、2、3、4、5。\n待评文本：\n"


def encode_text(tokenizer, text):
    return tokenizer(PREFIX + text, truncation=False)


class QwenGradeEnsemble:
    def __init__(self, base_path, adapters, tokenizer, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel

        self.device = device
        self.tokenizer = tokenizer
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=5,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        base.config.use_cache = False
        self.model = PeftModel.from_pretrained(
            base, adapters[0], adapter_name="seed0", is_trainable=False
        )
        self.names = ["seed0"]
        for i, path in enumerate(adapters[1:], 1):
            name = f"seed{i}"
            self.model.load_adapter(path, adapter_name=name, is_trainable=False)
            self.names.append(name)
        self.model.to(device).eval()

    def probabilities(self, texts):
        import torch

        encoded = [encode_text(self.tokenizer, t) for t in texts]
        if any(len(x["input_ids"]) > 8192 for x in encoded):
            raise ValueError("Qwen input exceeds 8192 tokens; no truncation")
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt").to(
            self.device
        )
        predictions = []
        with torch.inference_mode():
            for name in self.names:
                self.model.set_adapter(name)
                self.model.eval()
                logits = self.model(**batch).logits.float()
                predictions.append(logits.softmax(-1).cpu().numpy())
        return np.mean(predictions, axis=0)
