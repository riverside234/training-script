"""
dp+pp

deepspeed --num_gpus=6 train_deepspeed_simple_prof.py --deepspeed_config ds_stage0_prof.json

"""
from transformers import AutoTokenizer, Llama4ForConditionalGeneration, DataCollatorWithPadding
from lora import LoRALinear, add_lora_to_llama4
import deepspeed
import os
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
from transformers.integrations import HfDeepSpeedConfig
import json
from pipeline import EmbedPipe, BlockPipe, NormPipe, LMHeadPipe, causal_lm_loss
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from pathlib import Path
from datasets import load_dataset, Features, Value, Sequence, load_from_disk


#----------------ini deepspeed
if not dist.is_initialized():
    deepspeed.init_distributed()

ds_config = json.load(open("ds_config_0.json", "r"))

dschf = HfDeepSpeedConfig(ds_config) 
#----------------- Ini model
model_id = "unsloth/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    dtype="bfloat16",
    device_map=None,    
    trust_remote_code   = True,
    cache_dir="/scratch/user/u.yx314365/cache",
    low_cpu_mem_usage=True
)

model.config.use_cache = False

"""
#print layers
for name, module in model.named_modules():
    if "language_model.model.layers.0" in name:
        print(name, ":", module.__class__.__name__)
"""

#load lora layer to the model---------------------------

add_lora_to_llama4(model, 8, 8)

# only fine-tuning on lora parameter  
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad = True

#build pipeline model--------------------------
lm = model.language_model.model  
lm_head = model.get_output_embeddings()

layers = [
    EmbedPipe(lm.embed_tokens),
    *[BlockPipe(l) for l in lm.layers],
    NormPipe(lm.norm),
    LMHeadPipe(lm_head),
]

pipe = PipelineModule(
    layers=layers,
    loss_fn=causal_lm_loss,
    num_stages=3,
    partition_method="uniform",  
    activation_checkpoint_interval=1,
)
print("finished\n")

#-------------------training with profiler
trainable_params = [p for p in pipe.parameters() if p.requires_grad]
print(trainable_params)

optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
ds_config = "ds_config_0.json"

torch.cuda.empty_cache()

engine, optimizer, _, _ = deepspeed.initialize(
    model=pipe,
    model_parameters=trainable_params,
    config=ds_config,  
    optimizer=optimizer
)

#------------------------data batch prep
def to_features(ex):
    messages = ex["messages"]
        
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        chat_str,
        truncation=True,
        max_length=tok.model_max_length,
        padding=False,           
        add_special_tokens=True,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }

_base_collator = DataCollatorWithPadding(
               tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"
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

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
train_iter = iter(RepeatingLoader(train_dataloader))

#-----------------training with profiler  
flops_prof = FlopsProfiler(engine.module, ds_engine=engine)

PROFILE_AT_STEP = 1
PROFILE_DURATION = 2

for step in range(20):
    if step == PROFILE_AT_STEP:
        flops_prof.start_profile()
        engine.print_rank_0(f"=== START profile step {step} ===")

    loss = engine.train_batch(data_iter=train_iter)  

    if step == PROFILE_AT_STEP + PROFILE_DURATION - 1:
        flops_prof.stop_profile()
        engine.print_rank_0(f"=== STOP profile step {step} ===")
        flops_prof.print_model_profile(profile_step=PROFILE_AT_STEP, module_depth=-1, top_modules=10, detailed=True)
        break
