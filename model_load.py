import os, json, re
from safetensors import safe_open
import torch.distributed as dist

def _find_index_json(model_dir):
    for fn in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        path = os.path.join(model_dir, fn)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No HF index json found (expect model.safetensors.index.json).")

def load_local_hf_weights(engine, model_dir, cfg):
    index_path = _find_index_json(model_dir)
    index = json.load(open(index_path, "r"))
    weight_map = index["weight_map"]  # hf_key -> shard filename

    # detect how PipelineModule names layers in state_dict
    local_keys = list(engine.module.state_dict().keys())
    root = "layers" if any(k.startswith("layers.") for k in local_keys) else "forward_funcs"

    # choose lm_head prefix that exists in the checkpoint
    head_prefix_candidates = ["language_model.lm_head", "language_model.model.lm_head", "lm_head"]
    head_prefix = None
    for cand in head_prefix_candidates:
        if f"{cand}.weight" in weight_map:
            head_prefix = cand
            break
    if head_prefix is None:
        raise RuntimeError("Could not find lm_head weight key in checkpoint index.")

    def pipe_key_to_hf_key(k):
        # skip LoRA params
        if "lora_A" in k or "lora_B" in k:
            return None

        m = re.match(rf"^{root}\.(\d+)\.(.+)$", k)
        if not m:
            return None
        idx = int(m.group(1))
        rest = m.group(2)

        n = cfg.num_hidden_layers
        if idx == 0:
            return f"language_model.model.embed_tokens.{rest}"
        if 1 <= idx <= n:
            return f"language_model.model.layers.{idx-1}.{rest}"
        if idx == n + 1:
            return f"language_model.model.norm.{rest}"
        if idx == n + 2:
            return f"{head_prefix}.{rest}"
        return None

    # build mapping for just *this stage’s* parameters
    pipe_to_hf = {}
    for pk in local_keys:
        hk = pipe_key_to_hf_key(pk)
        if hk is not None and hk in weight_map:
            pipe_to_hf[pk] = hk

    # group needed hf keys by shard file
    by_file = {}
    for pk, hk in pipe_to_hf.items():
        fn = weight_map[hk]
        by_file.setdefault(fn, []).append((pk, hk))

    partial = {}
    for fn, pairs in by_file.items():
        shard_path = os.path.join(model_dir, fn)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for pk, hk in pairs:
                partial[pk] = f.get_tensor(hk)

    missing, unexpected = engine.module.load_state_dict(partial, strict=False)
    if dist.get_rank() == 0:
        print("Loaded stage-local weights. Missing (expected for LoRA):", len(missing), "Unexpected:", len(unexpected))

    dist.barrier()