from transformers import AutoConfig
from deepspeed.pipe import PipelineModule, LayerSpec
import torch.nn.functional as F
from lora import LoRALinear, inject_lora_attention
import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import Llama4TextDecoderLayer, Llama4TextRMSNorm
import torch

TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class EmbedPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=getattr(cfg, "pad_token_id", None))
        
    def forward(self, inputs):
        input_ids, attention_mask = inputs
        hidden = self.embed_tokens(input_ids)
        
        b, t = input_ids.shape
        position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)

        position_embeddings = build_rope_cos_sin(
            self.cfg, position_ids,
            device=input_ids.device,
            dtype=hidden.dtype
        )

        
        return (hidden, attention_mask, position_ids, position_embeddings)

class DecoderLayerPipe(Llama4TextDecoderLayer):
    def __init__(self, cfg, layer_idx, lora_r=0, lora_alpha=0, lora_dropout=0.0):
        super().__init__(cfg, layer_idx)
        self.layer_idx = layer_idx
        if lora_r and lora_r > 0:
            inject_lora_attention(self, suffixes=TARGET_SUFFIXES, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def forward(self, inputs):
        hidden, attention_mask, position_ids, position_embeddings = inputs

        if not layer_uses_rope(self.config, self.layer_idx):
            position_embeddings = None
            
        out = super().forward(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden = out[0] if isinstance(out, (tuple, list)) else out.hidden_states
        return (hidden, attention_mask, position_ids, position_embeddings)

class FinalNormPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = Llama4TextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, inputs):
        hidden, attention_mask, position_ids, position_embeddings = inputs
        return (self.norm(hidden), attention_mask, position_ids, position_embeddings)

class LMHeadPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, inputs):
        hidden, attention_mask, position_ids, position_embeddings = inputs
        return self.lm_head(hidden)

def causal_lm_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1),
                           ignore_index=-100)





#----------------------Position Embeddings
class PositionEmbeddings(tuple):
    """Tuple-like (cos, sin) that also supports `.to(...)`."""
    def to(self, device=None, dtype=None, non_blocking=False):
        return PositionEmbeddings(
            t.to(device=device, dtype=dtype, non_blocking=non_blocking) for t in self
        )

def build_rope_cos_sin(cfg, position_ids, *, device, dtype):
    # head_dim: prefer config fields if present
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    # partial rotary support (common in llama-family)
    partial = float(getattr(cfg, "partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    rotary_dim = rotary_dim - (rotary_dim % 2)

    base = float(getattr(cfg, "rope_theta", 10000.0))

    # (optional) rope scaling (approx; good enough for most training <= base context)
    pos = position_ids.to(torch.float32)
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and "factor" in rs:
        factor = float(rs["factor"])
        # linear-like scaling approximation
        pos = pos / factor

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    freqs = torch.einsum("bt,d->btd", pos, inv_freq)           # [b, t, rotary_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                    # [b, t, rotary_dim]
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)

    return PositionEmbeddings((cos, sin))

def layer_uses_rope(cfg, layer_idx: int) -> bool:
    no_rope_layers = getattr(cfg, "no_rope_layers", None)
    if isinstance(no_rope_layers, list) and len(no_rope_layers) > 0:
        return (layer_idx not in no_rope_layers)

    interval = getattr(cfg, "nope_layer_interval", None)
    if isinstance(interval, int) and interval > 0:
        # “every Nth layer” → idx  N-1, 2N-1, ...
        return ((layer_idx + 1) % interval) != 0

    return True