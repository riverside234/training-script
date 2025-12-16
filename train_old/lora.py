
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # frozen base weight + bias
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # LoRA bits
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base + self.scaling * delta


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def add_lora_to_llama4(module, r=8, alpha=8, target_suffixes=TARGET_SUFFIXES, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and full_name.endswith(target_suffixes):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            # Recurse
            add_lora_to_llama4(child, r=r, alpha=alpha, target_suffixes=target_suffixes, prefix=full_name)