#!/usr/bin/env python3
"""
Convert Chris GPT-2 model_19072.pt to GGUF using llama.cpp's official
GPT2LMHeadModel converter.

Pipeline:
    Chris .pt
      -> exact Hugging Face GPT2LMHeadModel SafeTensors conversion
      -> llama.cpp convert_hf_to_gguf.py
      -> optional llama-quantize (e.g. Q4_K_M)

CPU-only conversion is supported.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Chris GPT-2 .pt checkpoint to GGUF")
    p.add_argument("checkpoint", type=Path, help="Path to model_19072.pt")
    p.add_argument("output", type=Path, help="Output .gguf path")
    p.add_argument(
        "--llama-cpp",
        type=Path,
        required=True,
        help="Path to a current llama.cpp checkout",
    )
    p.add_argument(
        "--outtype",
        default="f16",
        help="llama.cpp convert_hf_to_gguf.py outtype (default: f16; current examples include f32/f16/bf16/q8_0/auto)",
    )
    p.add_argument(
        "--quantize",
        default=None,
        help="Optional llama-quantize type, e.g. Q4_K_M, Q5_K_M, Q8_0. If set, an F16 GGUF is created first.",
    )
    p.add_argument(
        "--quantize-bin",
        type=Path,
        default=None,
        help="Explicit path to llama-quantize binary (auto-detected otherwise)",
    )
    p.add_argument(
        "--tokenizer-source",
        default="openai-community/gpt2",
        help="GPT-2 tokenizer source passed to the HF converter",
    )
    p.add_argument(
        "--hf-dir",
        type=Path,
        default=None,
        help="Keep/use the intermediate Hugging Face directory at this path",
    )
    p.add_argument(
        "--keep-intermediate-gguf",
        action="store_true",
        help="When quantizing, keep the intermediate F16 GGUF next to output",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output/intermediate HF directory when possible",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_quantize_binary(llama_cpp: Path) -> Path:
    names = ["llama-quantize", "quantize"]
    if sys.platform.startswith("win"):
        names = [n + ".exe" for n in names] + names

    candidates = []
    for name in names:
        candidates.extend([
            llama_cpp / "build" / "bin" / name,
            llama_cpp / "build" / "bin" / "Release" / name,
            llama_cpp / name,
        ])

    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "Could not find llama-quantize. Build llama.cpp first, or pass --quantize-bin.\n"
        "Typical Linux build: cmake -B build && cmake --build build -j"
    )


def convert_hf(checkpoint: Path, hf_dir: Path, tokenizer_source: str, overwrite: bool) -> None:
    helper = Path(__file__).with_name("convert_chris_gpt2_to_hf.py")
    if not helper.is_file():
        raise FileNotFoundError(
            f"Missing sibling converter: {helper}\n"
            "Keep convert_chris_gpt2_to_hf.py in the same directory as this script."
        )

    cmd = [
        sys.executable,
        str(helper),
        str(checkpoint),
        str(hf_dir),
        "--tokenizer-source",
        tokenizer_source,
    ]
    if overwrite:
        cmd.append("--overwrite")
    run(cmd)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    llama_cpp = args.llama_cpp.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.suffix.lower() != ".gguf":
        raise SystemExit("Output filename must end in .gguf")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}\nUse --overwrite to replace it.")

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        raise FileNotFoundError(f"Not found: {convert_script}")

    output.parent.mkdir(parents=True, exist_ok=True)

    temp_ctx = None
    if args.hf_dir is not None:
        hf_dir = args.hf_dir.resolve()
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="chris-gpt2-hf-")
        hf_dir = Path(temp_ctx.name)

    try:
        print("[1/3] Chris .pt -> Hugging Face")
        convert_hf(checkpoint, hf_dir, args.tokenizer_source, args.overwrite)

        if args.quantize:
            print("[2/3] Hugging Face -> F16 GGUF")
            intermediate = output.with_name(output.stem + ".f16-intermediate.gguf")
            if intermediate.exists():
                intermediate.unlink()

            run([
                sys.executable,
                str(convert_script),
                str(hf_dir),
                "--outfile",
                str(intermediate),
                "--outtype",
                "f16",
            ])

            print(f"[3/3] quantizing GGUF -> {args.quantize}")
            quant_bin = args.quantize_bin.resolve() if args.quantize_bin else find_quantize_binary(llama_cpp)
            if output.exists():
                output.unlink()
            run([str(quant_bin), str(intermediate), str(output), args.quantize])

            if not args.keep_intermediate_gguf:
                intermediate.unlink(missing_ok=True)
        else:
            print(f"[2/3] Hugging Face -> GGUF ({args.outtype})")
            if output.exists():
                output.unlink()
            run([
                sys.executable,
                str(convert_script),
                str(hf_dir),
                "--outfile",
                str(output),
                "--outtype",
                args.outtype,
            ])
            print("[3/3] no post-conversion quantization requested")

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"GGUF output was not created correctly: {output}")

        print("[ok] GGUF conversion complete")
        print(f"output: {output}")
        print(f"size:   {output.stat().st_size / (1024 ** 2):.2f} MiB")
        if args.hf_dir is not None:
            print(f"HF dir: {hf_dir}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()

