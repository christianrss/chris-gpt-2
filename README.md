# Chris-GPT-2

A from-scratch implementation and pretraining run of a **GPT-2 124M-scale causal language model**, trained on approximately **10 billion FineWeb-Edu tokens**.

The project covers the complete model lifecycle:

```text
dataset
   ↓
tokenization
   ↓
Transformer implementation
   ↓
distributed pretraining
   ↓
validation
   ↓
HellaSwag evaluation
   ↓
checkpointing
   ↓
inference
   ↓
Hugging Face conversion
   ↓
SafeTensors
   ↓
GGUF conversion
   ↓
custom inference runtimes
```

Chris-GPT-2 was trained **from randomly initialized weights**. It is not a fine-tune of the original OpenAI GPT-2 checkpoint.

---

## Public Artifacts

### Transformers / SafeTensors

Canonical Hugging Face model:

**https://huggingface.co/christianrss/chris-gpt-2-124m**

### GGUF

GGUF model repository:

**https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF**

### Original PyTorch Checkpoints

Native `.pt` training checkpoints:

**https://drive.google.com/drive/folders/1vD9DTvZKRBJjvY_ALqJWeXrVHYnZKOH6?usp=sharing**

### Technical Report

**A Reproducible 10-Billion-Token Pretraining Run of GPT-2 124M**

ResearchGate:

**https://www.researchgate.net/publication/412096883_A_Reproducible_10-Billion-Token_Pretraining_Run_of_GPT-2_124M**

---

# Model Details

| Property              |                Value |
| --------------------- | -------------------: |
| Architecture          |                GPT-2 |
| Parameters            |          124,475,904 |
| Transformer layers    |                   12 |
| Attention heads       |                   12 |
| Embedding dimension   |                  768 |
| Context length        |         1,024 tokens |
| Model vocabulary size |               50,304 |
| Tokenizer             | GPT-2 byte-level BPE |
| Tokenizer vocabulary  |               50,257 |
| Training tokens       |          ~10 billion |
| Dataset               |          FineWeb-Edu |
| Training              |         From scratch |
| Framework             |              PyTorch |
| Final validation loss |              3.07248 |

The model follows the GPT-2 124M architecture closely.

One implementation detail is the vocabulary size.

The standard GPT-2 tokenizer contains **50,257 tokens**, while Chris-GPT-2 uses a padded model vocabulary of **50,304 entries**.

```text
Tokenizer vocabulary: 50,257
Model vocabulary:     50,304
```

The additional model vocabulary entries were used as padding/alignment slots during training.

---

# Training

Chris-GPT-2 was pretrained from randomly initialized weights.

The target training budget was approximately 10 billion tokens:

```text
524,288 tokens/step × 19,073 steps
≈ 10,000,000,000 tokens
```

The final training run reached step:

```text
19,072
```

with a final validation loss of:

```text
3.0724804401397705
```

---

## Training Hardware

The main pretraining run used:

```text
4× NVIDIA A100 PCIe 40 GB
```

The training stack included:

* PyTorch
* Distributed Data Parallel
* NCCL
* mixed-precision training
* fused AdamW where available
* GPT-2 byte-level BPE
* FineWeb-Edu
* validation during training
* HellaSwag evaluation
* periodic checkpointing

---

# Dataset

Chris-GPT-2 was trained on approximately **10 billion tokens from FineWeb-Edu**.

FineWeb-Edu is an educationally filtered subset of FineWeb designed for language-model pretraining.

The training pipeline used the standard GPT-2 byte-level BPE tokenizer.

The token budget can be reproduced from the effective global batch:

```text
524,288 tokens per optimizer step
×
19,073 optimizer steps
≈
10 billion tokens
```

---

# Training Progress

Early in the run, the loss decreased from approximately:

```text
10.95
```

to:

```text
7.61
```

during the initial training phase.

The final validation loss was:

```text
3.0724804401397705
```

The experiment was designed around a fixed ~10-billion-token budget rather than reproducing the original OpenAI GPT-2 training corpus exactly.

---

# Original PyTorch Checkpoints

The original training checkpoints are preserved separately from the Hugging Face and GGUF releases.

Google Drive:

**https://drive.google.com/drive/folders/1vD9DTvZKRBJjvY_ALqJWeXrVHYnZKOH6?usp=sharing**

These `.pt` files are native artifacts produced directly by the Chris-GPT-2 training pipeline before conversion to any distribution format.

The final checkpoint is:

```text
model_19072.pt
```

The checkpoint contains more than model weights.

The stored training state includes:

```text
model
optimizer
config
step
val_loss
train_loader_current_shard
train_loader_current_position
torch_rng_state
cuda_rng_state
```

This makes the original checkpoints useful for:

* inspecting the native training state;
* reproducing the conversion process;
* studying checkpoint structure;
* validating the Hugging Face export;
* generating new GGUF exports;
* resuming or analyzing training;
* archival and reproducibility purposes.

For normal inference, the Hugging Face SafeTensors release is recommended.

---

# Hugging Face Model

The canonical model release is available at:

**https://huggingface.co/christianrss/chris-gpt-2-124m**

The model can be loaded directly using the standard Transformers API.

## Install

```bash
pip install torch transformers safetensors
```

## Load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "christianrss/chris-gpt-2-124m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

## Generate Text

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "christianrss/chris-gpt-2-124m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The future of artificial intelligence is"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
)

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )
)
```

---

# Example Generation

Prompt:

```text
The future of artificial intelligence is
```

Generation configuration:

```text
temperature = 0.8
top_k       = 50
top_p       = 0.95
max_new_tokens = 50
```

Actual sampled output:

```text
The future of artificial intelligence is a critical factor in the global economy. It’s also an important factor in our society’s ability to cope with global warming.
What are the future of artificial intelligence?
Artificial intelligence has the potential to become an important component
```

The output above was generated from the converted model and was not manually curated.

---

# Converting the Original Checkpoint to Hugging Face

The repository includes:

```text
convert_chris_gpt2_to_hf.py
```

This script converts a native Chris-GPT-2 `.pt` checkpoint into a standard Hugging Face `GPT2LMHeadModel` directory.

## Install Dependencies

```bash
pip install torch transformers safetensors tokenizers
```

## Convert

```bash
python convert_chris_gpt2_to_hf.py \
    /path/to/model_19072.pt \
    ./Chris-GPT-2-124M-HF
```

If the output directory already exists:

```bash
python convert_chris_gpt2_to_hf.py \
    /path/to/model_19072.pt \
    ./Chris-GPT-2-124M-HF \
    --overwrite
```

The resulting directory contains:

```text
Chris-GPT-2-124M-HF/
├── chris_conversion.json
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

---

# Why Weight Conversion Is Required

The original Chris-GPT-2 implementation uses `torch.nn.Linear` for the GPT-2 projection layers.

Hugging Face GPT-2 represents the corresponding layers with its `Conv1D` implementation.

Because the internal matrix layouts differ, the following weight matrices must be transposed during conversion:

```text
attn.c_attn.weight
attn.c_proj.weight
mlp.c_fc.weight
mlp.c_proj.weight
```

The conversion script performs this transformation automatically.

The converter also performs tensor-level checks against the original checkpoint.

After export, it reloads the model with:

```python
GPT2LMHeadModel.from_pretrained(...)
```

and executes a deterministic forward pass to verify that:

* the model can be loaded;
* tensor shapes are correct;
* the output dimensions are correct;
* logits are finite.

---

# Validate the Local Hugging Face Export

From inside the generated Hugging Face directory:

```bash
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "."

tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
)

print("Tokenizer vocab:", len(tokenizer))
print("Model vocab:", model.config.vocab_size)

ids = tokenizer.encode("Hello world")

print("Token IDs:", ids)
print("Decoded:", tokenizer.decode(ids))
print("OK")
PY
```

Expected vocabulary configuration:

```text
Tokenizer vocab: 50257
Model vocab:     50304
```

The difference is intentional.

---

# Convert to GGUF

The repository also includes:

```text
convert_chris_gpt2_to_gguf.py
```

The conversion pipeline is:

```text
model_19072.pt
        │
        ▼
convert_chris_gpt2_to_hf.py
        │
        ▼
Hugging Face GPT2LMHeadModel
        │
        ▼
model.safetensors
        │
        ▼
llama.cpp convert_hf_to_gguf.py
        │
        ▼
Chris-GPT-2-124M-F16.gguf
```

The conversion intentionally passes through the validated Hugging Face representation instead of converting the native checkpoint directly into GGUF.

---

# Install llama.cpp

Clone llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

Enter the repository:

```bash
cd llama.cpp
```

Install the Python dependencies required by the current converter.

Depending on the llama.cpp version, this may be:

```bash
pip install -r requirements.txt
```

or another requirements file documented by the current llama.cpp repository.

---

# Convert Directly to F16 GGUF

Keep both converters in the same directory:

```text
convert_chris_gpt2_to_hf.py
convert_chris_gpt2_to_gguf.py
```

Run:

```bash
python convert_chris_gpt2_to_gguf.py \
    /path/to/model_19072.pt \
    Chris-GPT-2-124M-F16.gguf \
    --llama-cpp ~/llama.cpp
```

The default GGUF output type is:

```text
f16
```

The resulting file is:

```text
Chris-GPT-2-124M-F16.gguf
```

---

# Keep the Intermediate Hugging Face Export

By default, `convert_chris_gpt2_to_gguf.py` creates the Hugging Face model inside a temporary directory.

That temporary directory is removed after GGUF conversion.

To preserve the Hugging Face model, use:

```bash
python convert_chris_gpt2_to_gguf.py \
    /path/to/model_19072.pt \
    Chris-GPT-2-124M-F16.gguf \
    --llama-cpp ~/llama.cpp \
    --hf-dir ./Chris-GPT-2-124M-HF \
    --overwrite
```

This produces:

```text
Chris-GPT-2-124M-HF/
```

and:

```text
Chris-GPT-2-124M-F16.gguf
```

---

# GGUF Quantization

The GGUF converter can optionally invoke the llama.cpp quantizer.

Build llama.cpp:

```bash
cd ~/llama.cpp

cmake -B build
cmake --build build -j
```

The quantization executable is normally generated at:

```text
build/bin/llama-quantize
```

Example Q8 conversion:

```bash
python convert_chris_gpt2_to_gguf.py \
    /path/to/model_19072.pt \
    Chris-GPT-2-124M-Q8_0.gguf \
    --llama-cpp ~/llama.cpp \
    --quantize Q8_0
```

The script first creates an F16 GGUF and then passes it to `llama-quantize`.

To preserve the intermediate F16 model:

```bash
python convert_chris_gpt2_to_gguf.py \
    /path/to/model_19072.pt \
    Chris-GPT-2-124M-Q8_0.gguf \
    --llama-cpp ~/llama.cpp \
    --quantize Q8_0 \
    --keep-intermediate-gguf
```

Other quantization types depend on the llama.cpp version and architecture support.

Any quantized model should be validated before publication.

---

# GGUF Release

The public GGUF repository is:

**https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF**

The reference export is:

```text
Chris-GPT-2-124M-F16.gguf
```

The GGUF representation is intended for:

* custom inference runtimes;
* GGUF tooling;
* quantization experiments;
* model-format research;
* low-level systems work;
* Chris Llama development.

For standard Transformers inference, use the SafeTensors model instead.

---

# Publishing the Hugging Face Model

Authenticate:

```bash
hf auth login
```

Upload the Hugging Face directory:

```bash
hf upload \
    christianrss/chris-gpt-2-124m \
    ./Chris-GPT-2-124M-HF \
    .
```

The canonical repository is:

```text
christianrss/chris-gpt-2-124m
```

---

# Publishing GGUF

Upload an F16 GGUF file with:

```bash
hf upload \
    christianrss/chris-gpt-2-124m-GGUF \
    ./Chris-GPT-2-124M-F16.gguf \
    Chris-GPT-2-124M-F16.gguf
```

Additional validated quantizations can be added to the same repository.

---

# Model Distribution

Chris-GPT-2 is intentionally preserved in multiple representations.

```text
                    Chris-GPT-2
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    PyTorch .pt     SafeTensors       GGUF
          │              │              │
          ▼              ▼              ▼
     Google Drive   Hugging Face   Hugging Face
                      Transformers      GGUF
```

Each representation serves a different purpose.

## PyTorch Checkpoints

Google Drive:

**https://drive.google.com/drive/folders/1vD9DTvZKRBJjvY_ALqJWeXrVHYnZKOH6?usp=sharing**

Useful for:

* original training state;
* reproducibility;
* conversion;
* checkpoint analysis;
* training continuation.

## SafeTensors

Hugging Face:

**https://huggingface.co/christianrss/chris-gpt-2-124m**

Useful for:

* Transformers;
* PyTorch inference;
* evaluation;
* experimentation;
* fine-tuning.

## GGUF

Hugging Face:

**https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF**

Useful for:

* custom runtimes;
* quantization;
* low-level inference;
* GGUF tooling;
* Chris Llama.

---

# Reproducibility

The complete artifact chain is:

```text
FineWeb-Edu
      │
      ▼
Chris-GPT-2 training code
      │
      ▼
PyTorch checkpoints (.pt)
      │
      ├──────────────► Google Drive archive
      │
      ▼
model_19072.pt
      │
      ▼
Hugging Face conversion
      │
      ▼
model.safetensors
      │
      ├──────────────► Hugging Face Transformers
      │
      ▼
GGUF conversion
      │
      ▼
Chris-GPT-2-124M-F16.gguf
      │
      ├──────────────► Hugging Face GGUF
      │
      ▼
custom inference runtimes
```

The corresponding public resources are:

```text
Source code       → GitHub
Technical report  → ResearchGate
Training state    → Google Drive
SafeTensors       → Hugging Face
GGUF              → Hugging Face
```

---

# Project Structure

The repository is centered around the training and conversion pipeline.

A typical layout is:

```text
chris-gpt-2/
│
├── train_gpt2.py
├── hellaswag.py
├── convert_chris_gpt2_to_hf.py
├── convert_chris_gpt2_to_gguf.py
├── README.md
└── ...
```

Main components:

### `train_gpt2.py`

GPT-2 architecture, training loop, distributed execution, validation, checkpointing, and generation.

### `hellaswag.py`

Utilities used for HellaSwag evaluation.

### `convert_chris_gpt2_to_hf.py`

Converts the native Chris-GPT-2 checkpoint into a standard Hugging Face GPT-2 model.

### `convert_chris_gpt2_to_gguf.py`

Runs the complete:

```text
Chris checkpoint → Hugging Face → GGUF
```

pipeline.

---

# Chris Llama

Chris-GPT-2 is also used as a reference workload for **Chris Llama**:

**https://github.com/christianrss/chris-llama**

Chris Llama is an experimental low-level LLM inference runtime used to study:

* GGUF parsing;
* tensor loading;
* GPT-2 inference;
* attention;
* KV caching;
* sampling;
* quantization;
* CPU execution;
* heterogeneous compute;
* AdaptiveCpp;
* SYCL;
* GPU acceleration.

The goal is to understand the systems involved in language-model inference rather than treating existing inference engines as black boxes.

The Chris-GPT-2 GGUF file provides a model produced within the same project ecosystem for testing that runtime.

---

# Chris Torch

**Chris Torch** is an experimental machine-learning framework:

**https://github.com/christianrss/chris-torch**

The project explores lower-level implementations of concepts commonly provided by larger frameworks:

* tensors;
* autograd;
* modules;
* parameters;
* linear layers;
* optimizers;
* training loops;
* native compute kernels;
* C++ integration;
* SYCL;
* AdaptiveCpp.

Chris-GPT-2 provides a larger Transformer workload that can be used to study how such framework components relate to real model training.

---

# Relationship Between the Projects

The projects explore different parts of the machine-learning stack.

```text
                  Chris-GPT-2
                model + training
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
      Chris Torch             Chris Llama
  framework/training        inference/runtime
    foundations               foundations
           │                       │
           └───────────┬───────────┘
                       ▼
               heterogeneous
                  compute
                       │
               AdaptiveCpp / SYCL
```

The projects remain independent, but they share a common goal: understanding machine-learning systems below high-level APIs.

---

# Research

The main pretraining experiment is documented in:

## A Reproducible 10-Billion-Token Pretraining Run of GPT-2 124M

**Christian Rafael de Souza Silva**

ResearchGate:

**https://www.researchgate.net/publication/412096883_A_Reproducible_10-Billion-Token_Pretraining_Run_of_GPT-2_124M**

The report covers:

* GPT-2 reproduction;
* model architecture;
* FineWeb-Edu;
* token budget;
* distributed training;
* training dynamics;
* validation;
* HellaSwag evaluation;
* generation behavior;
* compute infrastructure;
* training cost;
* reproducibility.

The public model weights and original checkpoints provide the resulting artifacts from the experiment documented in the report.

---

# Public Resources

## Source Code

GitHub:

**https://github.com/christianrss/chris-gpt-2**

## Original PyTorch Checkpoints

Google Drive:

**https://drive.google.com/drive/folders/1vD9DTvZKRBJjvY_ALqJWeXrVHYnZKOH6?usp=sharing**

## Transformers / SafeTensors

Hugging Face:

**https://huggingface.co/christianrss/chris-gpt-2-124m**

## GGUF

Hugging Face:

**https://huggingface.co/christianrss/chris-gpt-2-124m-GGUF**

## Technical Report

ResearchGate:

**https://www.researchgate.net/publication/412096883_A_Reproducible_10-Billion-Token_Pretraining_Run_of_GPT-2_124M**

---

# Limitations

Chris-GPT-2 is a **base autoregressive language model**, not an instruction-tuned assistant.

It has not undergone:

* instruction tuning;
* reinforcement learning from human feedback;
* preference optimization;
* conversational alignment;
* safety fine-tuning.

The model can produce:

* incorrect information;
* repetition;
* inconsistent reasoning;
* incoherent continuations;
* biased or undesirable text.

It should not be used as an authoritative factual source.

The primary purpose of the model is:

* research;
* engineering;
* reproducibility;
* language-model systems experimentation;
* training infrastructure experiments;
* inference runtime development.

---

# Why GPT-2 124M?

GPT-2 124M is small enough to train and inspect using accessible accelerator infrastructure while still containing the fundamental components of modern autoregressive Transformers:

```text
token embeddings
position embeddings
multi-head self-attention
causal masking
residual connections
layer normalization
feed-forward networks
language-model head
autoregressive generation
```

This makes it useful for studying the full model lifecycle without hiding the implementation behind a much larger system.

---

# References

* Radford, A. et al. **Language Models are Unsupervised Multitask Learners.** OpenAI, 2019.
* Vaswani, A. et al. **Attention Is All You Need.** arXiv:1706.03762, 2017.
* Hugging Face FineWeb / FineWeb-Edu.
* Hugging Face Transformers.
* GGUF / GGML.
* llama.cpp.
* Karpathy, A. **Let's reproduce GPT-2 (124M)**.

---

# Author

**Christian Rafael de Souza Silva**

Independent Researcher

GitHub:
**https://github.com/christianrss**

Hugging Face:
**https://huggingface.co/christianrss**

ResearchGate:
**https://www.researchgate.net/publication/412096883_A_Reproducible_10-Billion-Token_Pretraining_Run_of_GPT-2_124M**
