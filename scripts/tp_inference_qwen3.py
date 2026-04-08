#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan CPU Inference for HuggingFace Qwen3-0.6B

Usage:
    python scripts/tp_inference_qwen3.py --model_name Qwen/Qwen3-0.6B --prompt "Hello"

Environment:
    1. Create conda env: conda create -n titan python=3.10 -y && conda activate titan
    2. Install torch: pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    3. Install transformers: pip install transformers
    4. Download model: huggingface-cli download Qwen/Qwen3-0.6B
"""

import argparse
import os
import time
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


@torch.no_grad()
def generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """自回归生成"""
    past_key_values = None
    generated = input_ids

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated[:, -1:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits  # (batch, 1, vocab)
        past_key_values = outputs.past_key_values

        # 取最后一个位置的 logit
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


def main():
    parser = argparse.ArgumentParser(description="TorchTitan CPU Inference for Qwen3-0.6B")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path"
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

    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"TorchTitan CPU Inference for Qwen3")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"{'='*60}\n")

    # ===== Load Tokenizer =====
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
        print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
        eos_token_id = None

    # ===== Load Config =====
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
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
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        print("Pretrained model loaded successfully.")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("Creating randomly initialized model.")
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM(config)
        if dtype != torch.float32:
            model = model.to(dtype)

    model = model.to(device)
    model.eval()

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
    start_time = time.time()

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    elapsed = time.time() - start_time

    # ===== Print Results =====
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
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
    print("\nDone.")


if __name__ == "__main__":
    main()
