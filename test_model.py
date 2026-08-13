import argparse
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# Constants
# ============================================================

GPT2_TOKENIZER_VOCAB_SIZE = 50257


# ============================================================
# Configuration
# ============================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# ============================================================
# Model
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # Query, key and value projection
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        # Output projection
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # This buffer existed in the training model.
        #
        # The current forward() uses PyTorch SDPA with
        # is_causal=True, so this tensor is not needed
        # for the actual attention calculation.
        #
        # However, it MUST exist here because it is part
        # of the checkpoint state_dict.
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):

        B, T, C = x.size()

        # Generate Q, K and V together
        qkv = self.c_attn(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=2
        )

        head_size = C // self.n_head

        # (B, T, C)
        # ->
        # (B, nh, T, hs)

        q = q.view(
            B,
            T,
            self.n_head,
            head_size
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            head_size
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            head_size
        ).transpose(1, 2)

        # Flash / SDPA attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        # Reassemble all heads
        y = (
            y
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # Output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.gelu = nn.GELU(
            approximate="tanh"
        )

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.n_embd
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln_2 = nn.LayerNorm(
            config.n_embd
        )

        self.mlp = MLP(
            config
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(

                # Token embeddings
                wte=nn.Embedding(
                    config.vocab_size,
                    config.n_embd
                ),

                # Position embeddings
                wpe=nn.Embedding(
                    config.block_size,
                    config.n_embd
                ),

                # Transformer blocks
                h=nn.ModuleList(
                    [
                        Block(config)
                        for _ in range(
                            config.n_layer
                        )
                    ]
                ),

                # Final LayerNorm
                ln_f=nn.LayerNorm(
                    config.n_embd
                ),
            )
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.transformer.wte.weight = (
            self.lm_head.weight
        )

    def forward(self, idx):

        B, T = idx.size()

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block size {self.config.block_size}"
            )

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path, device):

    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    config = checkpoint["config"]

    print(f"checkpoint : {checkpoint_path}")
    print(f"step       : {checkpoint.get('step')}")
    print(f"val loss   : {checkpoint.get('val_loss')}")
    print(f"block size : {config.block_size}")
    print(f"vocab size : {config.vocab_size}")
    print(f"layers     : {config.n_layer}")
    print(f"heads      : {config.n_head}")
    print(f"embedding  : {config.n_embd}")

    model = GPT(config)

    # strict=True is intentional.
    # We want the evaluation model to match
    # the training checkpoint exactly.
    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.to(device)
    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"parameters : "
        f"{parameter_count:,}"
    )

    print(
        f"state dict : "
        f"{len(checkpoint['model'])} tensors"
    )

    print(f"device     : {device}")

    print()
    print("Checkpoint loaded successfully.")
    print()

    return model, checkpoint


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate(
    model,
    enc,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    seed=42,
    greedy=False
):

    prompt_tokens = enc.encode(prompt)

    x = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(seed)

    generated_tokens = []

    start_time = time.perf_counter()

    for _ in range(max_new_tokens):

        # GPT-2 only supports 1024 tokens
        x_cond = x[
            :,
            -model.config.block_size:
        ]

        logits = model(x_cond)

        # Only final token prediction
        logits = logits[:, -1, :]

        # IMPORTANT
        #
        # Training model:
        # vocab_size = 50304
        #
        # GPT-2 tokenizer:
        # vocab_size = 50257
        #
        # IDs >= 50257 cannot be decoded by
        # the GPT-2 tokenizer.
        logits = logits[
            :,
            :GPT2_TOKENIZER_VOCAB_SIZE
        ]

        if greedy:

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        else:

            temp = max(
                temperature,
                1e-5
            )

            logits = logits / temp

            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                top_values, _ = torch.topk(
                    logits,
                    k
                )

                cutoff = top_values[:, [-1]]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf")
                )

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

        token_id = next_token.item()

        generated_tokens.append(
            token_id
        )

        x = torch.cat(
            (
                x,
                next_token
            ),
            dim=1
        )

        # <|endoftext|>
        if token_id == 50256:
            break

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    completion = enc.decode(
        generated_tokens
    )

    if elapsed > 0:

        tokens_per_second = (
            len(generated_tokens)
            / elapsed
        )

    else:

        tokens_per_second = 0.0

    return {
        "prompt": prompt,
        "completion": completion,

        "tokens_generated":
            len(generated_tokens),

        "seconds":
            elapsed,

        "tokens_per_second":
            tokens_per_second,

        "temperature":
            temperature,

        "top_k":
            top_k,

        "seed":
            seed,

        "greedy":
            greedy,
    }


# ============================================================
# Test prompts
# ============================================================

TESTS = {

    "basic_language": [

        "The meaning of life is",

        "The most important thing to remember is",

        "In the modern world, technology has",

        "One of the biggest challenges facing humanity is",
    ],


    "encyclopedic": [

        "The Earth revolves around the Sun because",

        "Albert Einstein was a physicist who",

        "The history of the Roman Empire",

        "The human brain is",

        "The Pacific Ocean is",
    ],


    "science": [

        "Machine learning is a field of computer science that",

        "A neural network learns by",

        "The theory of relativity explains",

        "DNA contains",

        "Gravity is a force that",
    ],


    "programming": [

        "Python is a programming language that",

        "In computer programming, a function is",

        "A hash table is a data structure that",

        "The purpose of an operating system is",

        "A compiler translates",
    ],


    "code": [

        "def fibonacci(n):\n",

        "def add(a, b):\n",

        "# Python program to calculate the factorial of a number\n",

        "for i in range(10):\n",
    ],


    "reasoning_style": [

        "If John has five apples and gives two apples away, then",

        "There are 10 people in a room. If 3 people leave,",

        "A train travels at 60 miles per hour for two hours.",

        "If all dogs are animals and Rex is a dog, then",
    ],


    "story": [

        "Once upon a time, in a small village near the mountains,",

        "The old scientist opened the laboratory door and discovered",

        "It was midnight when Sarah heard a strange sound outside.",

        "The spaceship landed on the unknown planet and",
    ],


    "long_form": [

        (
            "Artificial intelligence has changed dramatically "
            "over the last several decades. Early researchers "
            "focused on symbolic reasoning, while modern systems"
        ),

        (
            "Computer processors have become dramatically faster "
            "over time, but performance depends on much more than "
            "clock frequency. Modern processors"
        ),

        (
            "The development of the internet transformed the way "
            "people communicate, work, and access information. "
            "One important consequence of this transformation is"
        ),
    ],


    "question_answer": [

        "Question: What is the capital of France?\nAnswer:",

        "Question: What planet is known as the Red Planet?\nAnswer:",

        "Question: What is photosynthesis?\nAnswer:",

        "Question: Who wrote Romeo and Juliet?\nAnswer:",

        "Question: What is the largest ocean on Earth?\nAnswer:",
    ],
}


# ============================================================
# Text metrics
# ============================================================

def repetition_score(text):

    words = text.lower().split()

    if len(words) < 2:
        return 0.0

    bigrams = []

    for i in range(
        len(words) - 1
    ):

        bigrams.append(
            (
                words[i],
                words[i + 1]
            )
        )

    if len(bigrams) == 0:
        return 0.0

    unique_bigrams = len(
        set(bigrams)
    )

    repetition = (
        1.0
        - unique_bigrams
        / len(bigrams)
    )

    return repetition


# ============================================================
# Full suite
# ============================================================

def run_suite(
    model,
    enc,
    device,
    max_new_tokens,
    output
):

    results = []

    print()
    print("=" * 70)
    print("CHRIS GPT TEST SUITE")
    print("=" * 70)

    total_tests = sum(
        len(prompts)
        for prompts in TESTS.values()
    )

    current_test = 0

    for category, prompts in TESTS.items():

        print()
        print("#" * 70)
        print(
            f"CATEGORY: "
            f"{category.upper()}"
        )
        print("#" * 70)

        for prompt in prompts:

            current_test += 1

            print()
            print(
                f"TEST "
                f"{current_test}/{total_tests}"
            )

            result = generate(
                model=model,
                enc=enc,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                seed=42
            )

            repetition = (
                repetition_score(
                    result["completion"]
                )
            )

            result[
                "category"
            ] = category

            result[
                "repetition_score"
            ] = repetition

            results.append(
                result
            )

            print()
            print("PROMPT")
            print("-" * 70)
            print(prompt)

            print()
            print("COMPLETION")
            print("-" * 70)

            print(
                result["completion"]
            )

            print()
            print(
                f"tokens generated : "
                f"{result['tokens_generated']}"
            )

            print(
                f"speed            : "
                f"{result['tokens_per_second']:.2f} tok/s"
            )

            print(
                f"repetition       : "
                f"{repetition:.4f}"
            )

    # Summary
    speeds = [
        r["tokens_per_second"]
        for r in results
    ]

    repetitions = [
        r["repetition_score"]
        for r in results
    ]

    summary = {

        "tests":
            len(results),

        "average_tokens_per_second":
            (
                sum(speeds)
                / len(speeds)
            ),

        "average_repetition_score":
            (
                sum(repetitions)
                / len(repetitions)
            ),
    }

    output_data = {

        "summary":
            summary,

        "results":
            results,
    }

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"tests               : "
        f"{summary['tests']}"
    )

    print(
        f"average speed       : "
        f"{summary['average_tokens_per_second']:.2f} tok/s"
    )

    print(
        f"average repetition  : "
        f"{summary['average_repetition_score']:.4f}"
    )

    print()
    print(
        f"results saved to: "
        f"{output}"
    )


# ============================================================
# Temperature test
# ============================================================

def temperature_test(
    model,
    enc,
    device,
    prompt
):

    temperatures = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
    ]

    print()
    print("=" * 70)
    print("TEMPERATURE TEST")
    print("=" * 70)

    print()
    print("PROMPT:")
    print(prompt)

    for temperature in temperatures:

        result = generate(
            model=model,
            enc=enc,
            prompt=prompt,
            device=device,
            max_new_tokens=120,
            temperature=temperature,
            top_k=50,
            seed=42
        )

        print()
        print("-" * 70)

        print(
            f"TEMPERATURE = "
            f"{temperature}"
        )

        print("-" * 70)

        print(
            result["completion"]
        )


# ============================================================
# Seed diversity test
# ============================================================

def seed_test(
    model,
    enc,
    device,
    prompt
):

    seeds = [
        1,
        42,
        1337,
        2026,
    ]

    print()
    print("=" * 70)
    print("SEED DIVERSITY TEST")
    print("=" * 70)

    print()
    print("PROMPT:")
    print(prompt)

    for seed in seeds:

        result = generate(
            model=model,
            enc=enc,
            prompt=prompt,
            device=device,
            max_new_tokens=120,
            temperature=0.8,
            top_k=50,
            seed=seed
        )

        print()
        print("-" * 70)

        print(
            f"SEED = {seed}"
        )

        print("-" * 70)

        print(
            result["completion"]
        )


# ============================================================
# Greedy vs top-k
# ============================================================

def decoding_test(
    model,
    enc,
    device,
    prompt
):

    print()
    print("=" * 70)
    print("DECODING COMPARISON")
    print("=" * 70)

    print()
    print("PROMPT:")
    print(prompt)

    greedy_result = generate(
        model=model,
        enc=enc,
        prompt=prompt,
        device=device,
        max_new_tokens=120,
        greedy=True
    )

    sampling_result = generate(
        model=model,
        enc=enc,
        prompt=prompt,
        device=device,
        max_new_tokens=120,
        temperature=0.8,
        top_k=50,
        seed=42
    )

    print()
    print("-" * 70)
    print("GREEDY")
    print("-" * 70)

    print(
        greedy_result["completion"]
    )

    print()
    print("-" * 70)
    print("TOP-K 50 / TEMP 0.8")
    print("-" * 70)

    print(
        sampling_result["completion"]
    )


# ============================================================
# Interactive generation
# ============================================================

def interactive(
    model,
    enc,
    device,
    tokens
):

    print()
    print("=" * 70)
    print("INTERACTIVE COMPLETION MODE")
    print("=" * 70)

    print(
        "Enter a text prompt and the "
        "model will continue it."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        print()

        prompt = input(
            "Prompt > "
        )

        if prompt.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        result = generate(
            model=model,
            enc=enc,
            prompt=prompt,
            device=device,
            max_new_tokens=tokens,
            temperature=0.8,
            top_k=50,
            seed=int(time.time())
        )

        print()
        print("-" * 70)

        print(
            prompt
            + result["completion"]
        )

        print("-" * 70)

        print(
            f"{result['tokens_generated']} tokens | "
            f"{result['tokens_per_second']:.2f} tok/s"
        )


# ============================================================
# Single prompt mode
# ============================================================

def single_prompt(
    model,
    enc,
    device,
    prompt,
    tokens
):

    result = generate(
        model=model,
        enc=enc,
        prompt=prompt,
        device=device,
        max_new_tokens=tokens,
        temperature=0.8,
        top_k=50,
        seed=42
    )

    print()
    print("=" * 70)
    print("PROMPT")
    print("=" * 70)

    print(prompt)

    print()
    print("=" * 70)
    print("COMPLETION")
    print("=" * 70)

    print(
        result["completion"]
    )

    print()
    print(
        f"tokens generated : "
        f"{result['tokens_generated']}"
    )

    print(
        f"speed            : "
        f"{result['tokens_per_second']:.2f} tok/s"
    )


# ============================================================
# Command line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Chris GPT-2 evaluation tool"
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="log/model_19072.pt",
        help="Checkpoint path"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "suite",
            "temperature",
            "seed",
            "decoding",
            "interactive",
            "prompt",
        ],
        default="suite"
    )

    parser.add_argument(
        "--prompt",
        default=(
            "Artificial intelligence is"
        )
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=120
    )

    parser.add_argument(
        "--output",
        default="model_test_results.json"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "cuda"

        print(
            f"CUDA GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        device = "mps"

    else:

        device = "cpu"

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    enc = tiktoken.get_encoding(
        "gpt2"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, checkpoint = load_model(
        args.checkpoint,
        device
    )

    # --------------------------------------------------------
    # Run selected mode
    # --------------------------------------------------------

    if args.mode == "suite":

        run_suite(
            model=model,
            enc=enc,
            device=device,
            max_new_tokens=args.tokens,
            output=args.output
        )

    elif args.mode == "temperature":

        temperature_test(
            model=model,
            enc=enc,
            device=device,
            prompt=args.prompt
        )

    elif args.mode == "seed":

        seed_test(
            model=model,
            enc=enc,
            device=device,
            prompt=args.prompt
        )

    elif args.mode == "decoding":

        decoding_test(
            model=model,
            enc=enc,
            device=device,
            prompt=args.prompt
        )

    elif args.mode == "interactive":

        interactive(
            model=model,
            enc=enc,
            device=device,
            tokens=args.tokens
        )

    elif args.mode == "prompt":

        single_prompt(
            model=model,
            enc=enc,
            device=device,
            prompt=args.prompt,
            tokens=args.tokens
        )


if __name__ == "__main__":
    main()