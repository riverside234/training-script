
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = (alpha / r) if r > 0 else 0.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

        # Freeze base, train LoRA
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            out = out + (F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scaling)
        return out

def _set_module_by_path(root: nn.Module, path: str, new_module: nn.Module):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)

def inject_lora_attention(root: nn.Module, *, suffixes, r=8, alpha=8, dropout=0.0, copy_base=True):
    """
    Walk `root` and replace nn.Linear whose full name endswith any suffix in `suffixes`.
    """
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suf) for suf in suffixes):
            continue

        new = LoRALinear(module.in_features, module.out_features,
                        bias=(module.bias is not None),
                        r=r, alpha=alpha, dropout=dropout).to(dtype=torch.bfloat16)

        if copy_base:
            with torch.no_grad():
                new.weight.copy_(module.weight)
                if module.bias is not None:
                    new.bias.copy_(module.bias)

        _set_module_by_path(root, name, new)


        
