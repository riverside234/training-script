from transformers import AutoConfig
from deepspeed.pipe import PipelineModule, LayerSpec
import torch.nn.functional as F
from lora import LoRALinear, inject_lora_attention
import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import Llama4TextDecoderLayer, Llama4TextRMSNorm, Llama4TextRotaryEmbedding
import torch

TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class EmbedPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=getattr(cfg, "pad_token_id", None))

    def forward(self, inputs):
        input_ids, attention_mask = inputs

        hidden = self.embed_tokens(input_ids)
        
        return (hidden, attention_mask)

class DecoderLayerPipe(Llama4TextDecoderLayer):
    def __init__(self, cfg, layer_idx, lora_r=0, lora_alpha=0, lora_dropout=0.0):
        super().__init__(cfg, layer_idx)
        self.rotary = Llama4TextRotaryEmbedding(cfg)

        if lora_r and lora_r > 0:
            inject_lora_attention(self, suffixes=TARGET_SUFFIXES, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        
    def forward(self, inputs):
        hidden, attention_mask_2d = inputs    
        
        hidden = hidden.contiguous()
        attention_mask_2d = attention_mask_2d.contiguous()

        # Build per-stage derived tensors  
        position_ids = build_position_ids(attention_mask_2d)             
        position_embeddings = self.rotary(hidden, position_ids)          

        pad_mask = (attention_mask_2d == 0)
        attn_mask_4d = pad_mask[:, None, None, :].contiguous()              #  contiguous

        cache_position = position_ids[0] if position_ids is not None else torch.arange(hidden.size(1), device=hidden.device)
        
        out = super().forward(
            hidden_states=hidden,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            use_cache=False,
        )

        return (out, attention_mask_2d)

class FinalNormPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = Llama4TextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, inputs):
        hidden, attention_mask_2d = inputs
        return (self.norm(hidden), attention_mask_2d)

class LMHeadPipe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, inputs):
        hidden, _attention_mask_2d = inputs
        return self.lm_head(hidden)

def causal_lm_loss(outputs, labels):
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs  # (bs, seq, vocab)

    min_seq = min(logits.size(1), labels.size(1))
    logits = logits[:, :min_seq, :]
    labels = labels[:, :min_seq]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid = shift_labels.ne(-100)
    denom = valid.sum()
    if denom.item() == 0:
        # keep graph; zero loss for this microbatch
        return shift_logits.sum() * 0.0

    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum / denom


def build_position_ids(attention_mask: torch.Tensor):

    # (bs, seq) with 1 for tokens, 0 for pad

    pos = attention_mask.long().cumsum(-1) - 1

    pos.masked_fill_(attention_mask == 0, 0)

    return pos

