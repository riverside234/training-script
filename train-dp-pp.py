"""
dp+pp

deepspeed --num_gpus=6 train_deepspeed_simple_prof.py --deepspeed_config ds_stage0_prof.json

"""
from transformers import AutoTokenizer, DataCollatorWithPadding
import deepspeed
import os
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
from transformers.integrations import HfDeepSpeedConfig
import json
from pipeline import EmbedPipe, DecoderLayerPipe, FinalNormPipe, LMHeadPipe, causal_lm_loss
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.utils import RepeatingLoader
from pathlib import Path
from datasets import load_dataset, Features, Value, Sequence, load_from_disk
from transformers import AutoConfig
from model_load import load_local_hf_weights
from torch.utils.data.distributed import DistributedSampler
from lora import LoRALinear
from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext
import deepspeed.comm as dist_deepspeed

#----------------ini deepspeed
deepspeed.init_distributed()

local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

#profiling for rank0 process, gpu0
is_rank0 = (dist.get_rank() == 0)  

prof_ctx = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    with_flops=True,
    with_modules=True,
    with_stack=True,
) if is_rank0 else nullcontext() 

#----------------- Ini model with pipeline
model_id = "unsloth/Llama-4-Scout-17B-16E-Instruct"
model_dir = "/scratch/user/u.yx314365/cache/model1"
cache_dir = "/scratch/user/u.yx314365/cache"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, cache_dir=cache_dir, model_max_length=4096)

cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, cache_dir=cache_dir)
cfg._attn_implementation = "sdpa" 

text_cfg = getattr(cfg, "text_config", None)
if text_cfg is None:
    text_cfg = getattr(cfg, "language_config", None)
if text_cfg is None:
    raise ValueError(f"Can't find text_config/language_config in {type(cfg)}")
    
n_layers = text_cfg.num_hidden_layers

specs = [
    LayerSpec(EmbedPipe, text_cfg),
    *[LayerSpec(DecoderLayerPipe, text_cfg, i, lora_r=4, lora_alpha=8, lora_dropout=0.05)
      for i in range(n_layers)],
    LayerSpec(FinalNormPipe, text_cfg),
    LayerSpec(LMHeadPipe, text_cfg),
]

old = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)

pipe = PipelineModule(
    layers=specs,
    loss_fn=causal_lm_loss,
    num_stages=4,
    partition_method="uniform",  
)

torch.set_default_dtype(old)

print("finished\n")
#-------------------ini training engine
for p in pipe.parameters():
    p.requires_grad = False

for m in pipe.modules():
    if isinstance(m, LoRALinear):
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)
        
trainable_params = [p for p in pipe.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(trainable_params, lr=1e-6)
ds_config = "ds_config_0.json"

engine, optimizer, _, _ = deepspeed.initialize(
    model=pipe,
    model_parameters=trainable_params,
    config=ds_config,  
    optimizer=optimizer
)

load_local_hf_weights(engine, model_dir, text_cfg)
dist.barrier()

#------------------------data batch prep
MAX_LEN = 2048
def to_features(ex):
    messages = ex["messages"]
        
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        chat_str,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,           
        add_special_tokens=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

_base_collator = DataCollatorWithPadding(
               tokenizer=tokenizer, padding="max_length",max_length=MAX_LEN, return_tensors="pt"
                )

def collate_fn(features):
    batch = _base_collator(features)  # dict of padded tensors

    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    inputs = (batch["input_ids"], batch["attention_mask"])
    return inputs, labels
    
main_path = Path("/scratch/user/u.yx314365")
data_path = main_path / f"content/drive/MyDrive/fed/tulu/clients-0.1/data0/mixed_train"
dataset = load_from_disk(str(data_path))

dataset = dataset.map(to_features, remove_columns=dataset.column_names, num_proc=2)    

sampler = DistributedSampler(dataset, shuffle=True)
train_dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, collate_fn=collate_fn)
train_iter = iter(RepeatingLoader(train_dataloader))

#-----------------training with profiler  
torch.cuda.reset_peak_memory_stats()

with prof_ctx as prof:
    for step in range(2):
        loss = engine.train_batch(data_iter=train_iter)
        if is_rank0:
            prof.step()

peak = torch.cuda.max_memory_allocated()

if is_rank0:
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=30))

t = torch.tensor([peak], device="cuda")
dist_deepspeed.all_reduce(t, op=dist_deepspeed.ReduceOp.MAX)
if dist_deepspeed.get_rank() == 0:
    print(f"Global peak max_memory_allocated: {t.item()/1024**3:.2f} GiB")
