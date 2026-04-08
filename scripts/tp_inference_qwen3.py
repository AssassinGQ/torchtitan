#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan TP Parallel Inference for HuggingFace Qwen3-0.6B

Usage:
    # Single-card CPU inference
    python scripts/tp_inference_qwen3.py --model_name Qwen/Qwen3-0.6B --prompt "Hello"

    # Multi-card CPU inference with torchrun
    torchrun --nproc_per_node=2 scripts/tp_inference_qwen3.py --model_name Qwen/Qwen3-0.6B --prompt "Hello"

Environment:
    1. Create conda env: conda create -n titan python=3.10 -y && conda activate titan
    2. Install torch: pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    3. Install transformers: pip install transformers
    4. Download model: python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')"
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.distributed.tensor import Shard, Replicate, DTensor
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model


def build_tp_plan_qwen3(model: Qwen3Model) -> dict:
    """为 Qwen3Model 构建 TP 并行策略"""
    num_layers = len(model.layers)
    plan = {}

    # embed_tokens: RowwiseParallel
    plan["embed_tokens"] = RowwiseParallel(output_layouts=Shard(1))

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}"

        # Self Attention: QKV Colwise, O Rowwise
        plan[f"{prefix}.self_attn.q_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.k_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.v_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()

        # MLP: gate/up Colwise, down Rowwise
        plan[f"{prefix}.mlp.gate_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.up_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()

    return plan


def apply_tp_to_model(model: Qwen3Model, tp_mesh: DeviceMesh) -> Qwen3Model:
    """应用 TP 并行到 Qwen3Model"""
    plan = build_tp_plan_qwen3(model)
    return parallelize_module(model, tp_mesh, plan)


def print_model_structure(model: torch.nn.Module, title: str, rank: int = 0):
    """打印模型结构和参数 shape"""
    if rank != 0:
        return
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    total_params = 0
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            local_shape = param.to_local().shape
            shape_str = " x ".join(str(s) for s in local_shape)
            num_params = param.to_local().numel()
            total_params += num_params
            print(f"  {name:<50} [{shape_str}] (global {param.shape}) {num_params:>12,} {param.placements}")
        else:
            shape_str = " x ".join(str(s) for s in param.shape)
            num_params = param.numel()
            total_params += num_params
            print(f"  {name:<50} [{shape_str}] {num_params:>12,}")
    print(f"{'='*70}")
    print(f"Total parameters (local per rank): {total_params:,}")
    print(f"{'='*70}\n")


def dtensor_to_local(tensor: torch.Tensor) -> torch.Tensor:
    """将 DTensor 转换回本地 tensor"""
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


@torch.no_grad()
def generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    use_tp: bool = False,
) -> torch.Tensor:
    """自回归生成"""
    past_key_values = None
    generated = input_ids

    for _ in range(max_new_tokens):
        if use_tp:
            # TP mode: use model.model with TP
            outputs = model.model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state
            # TP下需要all-gather
            last_hidden_state = dtensor_to_local(last_hidden_state)
            logits = model.lm_head(last_hidden_state)
            past_key_values = outputs.past_key_values
        else:
            # Non-TP mode: use model directly
            outputs = model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        next_token_logits = logits[:, -1, :] / temperature

        # Top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            sorted_indices_to_remove = cumsum > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        # Update attention mask
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones((attention_mask.shape[0], 1))
        ], dim=1) if attention_mask is not None else None

        # Check EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return generated


def broadcast_state_dict(state_dict: dict, src_rank: int = 0) -> dict:
    """Broadcast state_dict from src_rank to all other ranks"""
    if dist.get_world_size() == 1:
        return state_dict

    keys = list(state_dict.keys())
    key_tensor = torch.tensor([hash(k) for k in keys], dtype=torch.int64)
    dist.broadcast(key_tensor, src=src_rank)

    for key in keys:
        tensor = state_dict[key]
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.device.type == "meta":
            tensor = torch.empty_like(tensor, device="cpu")
        dist.broadcast(tensor, src=src_rank)
        state_dict[key] = tensor

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="TorchTitan TP Parallel Inference for Qwen3-0.6B")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor Parallel degree (default: 1, i.e., no TP)"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Input prompt (if not provided, will ask interactively)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Nucleus sampling top_p"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype (default: float32 for CPU)"
    )
    args = parser.parse_args()

    # Interactive prompt if not provided
    if args.prompt is None:
        args.prompt = input("Enter prompt: ").strip()
        if not args.prompt:
            args.prompt = "Hello, I'm a large language model."

    # ===== Distributed Init =====
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Determine if we're in a distributed context
    distributed_available = (rank >= 0 and world_size >= 0 and world_size > 1)

    if distributed_available:
        init_process_group(
            backend="gloo",  # CPU backend
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )
    else:
        # Single device mode
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cpu:{local_rank}" if world_size > 1 else "cpu")
    is_main = rank == 0

    # ===== Build Device Mesh for TP =====
    tp_degree = min(args.tp, world_size) if distributed_available else 1
    use_tp = distributed_available and tp_degree > 1

    if use_tp and tp_degree != args.tp and is_main:
        print(f"Warning: requested TP={args.tp} but using TP={tp_degree}")

    if use_tp:
        mesh = init_device_mesh(
            "cpu",
            (world_size,),
            mesh_dim_names=("tp",),
        )
        tp_mesh = mesh["tp"]
    else:
        tp_mesh = None

    # ===== Print Info =====
    if is_main:
        print(f"\n{'='*60}")
        print(f"TorchTitan TP Parallel Inference for Qwen3")
        print(f"{'='*60}")
        print(f"Model: {args.model_name}")
        print(f"TP degree: {tp_degree}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Prompt: {args.prompt}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print(f"{'='*60}\n")

    # ===== Load Tokenizer =====
    if is_main:
        print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.pad_token
        eos_token_id = tokenizer.eos_token_id
    except Exception as e:
        if is_main:
            print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
        eos_token_id = None

    # ===== Load Config =====
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if is_main:
        print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
              f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
              f"intermediate={config.intermediate_size}")

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # ===== Load Model =====
    if is_main:
        print("Loading model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if is_main:
            print("Pretrained model loaded successfully.")
    except Exception as e:
        if is_main:
            print(f"Error loading pretrained model: {e}")
            print("Creating randomly initialized model.")
        model = Qwen3ForCausalLM(config)
        if dtype != torch.float32:
            model = model.to(dtype)

    # Broadcast model to all ranks in distributed mode
    if distributed_available:
        state_dict = model.state_dict()
        state_dict = broadcast_state_dict(state_dict, src_rank=0)
        model.load_state_dict(state_dict, strict=False)
        del state_dict

    model = model.to(device)
    model.eval()

    # ===== Print model structure BEFORE TP =====
    print_model_structure(model, "Model Structure BEFORE TP Partitioning", rank)

    # ===== Apply TP Parallelism =====
    if use_tp:
        if is_main:
            print(f"Applying TP={tp_degree} to model...")
        model.model = apply_tp_to_model(model.model, tp_mesh)
        if distributed_available:
            dist.barrier()

        # ===== Print model structure AFTER TP =====
        print_model_structure(model, f"Model Structure AFTER TP-{tp_degree} Partitioning", rank)

    if is_main:
        print("Model ready. Starting inference...\n")

    # ===== Tokenize Input =====
    if tokenizer is not None:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    else:
        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
        attention_mask = None

    # ===== Run Inference =====
    if distributed_available:
        dist.barrier()

    start_time = time.time()

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        use_tp=use_tp,
    )

    elapsed = time.time() - start_time

    # ===== Print Results =====
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    if is_main:
        if tokenizer is not None:
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"\n{'='*60}")
            print(f"Generated text:")
            print(f"{'='*60}")
            print(generated_text)
            print(f"{'='*60}")

        print(f"Time: {elapsed:.2f}s")
        print(f"New tokens: {num_new_tokens}")
        print(f"Speed: {num_new_tokens / elapsed:.2f} tok/s")

    if distributed_available:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        print("\nDone.")


if __name__ == "__main__":
    main()
