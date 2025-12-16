import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.pipe import PipelineModule

class EmbedPipe(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, inputs):
        input_ids, attention_mask = inputs               # inputs come from dataloader
        hidden = self.embed_tokens(input_ids)

        # compute position_ids once, pass along
        b, t = input_ids.shape
        position_ids = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)

        return (hidden, attention_mask, position_ids)

class BlockPipe(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs):
        hidden, attention_mask, position_ids = inputs
        out = self.block(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = out[0] if isinstance(out, (tuple, list)) else out.hidden_states
        return (hidden, attention_mask, position_ids)

class NormPipe(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, inputs):
        hidden, attention_mask, position_ids = inputs
        return (self.norm(hidden), attention_mask, position_ids)

class LMHeadPipe(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, inputs):
        hidden, attention_mask, position_ids = inputs
        return self.lm_head(hidden)   # logits [B,T,V]

def causal_lm_loss(logits, labels):
    # shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )