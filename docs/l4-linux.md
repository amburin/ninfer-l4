# NVIDIA L4 Linux / Docker guide

This document describes the currently validated NInfer setup for a single NVIDIA L4 24 GB GPU.

## Hardware target

```text
GPU: NVIDIA L4
VRAM: 24 GB
CUDA architecture: sm_89
```

This fork must be built with:

```text
CMAKE_CUDA_ARCHITECTURES=89
```

## Host requirements

The host only needs:

- NVIDIA driver
- Docker
- NVIDIA Container Toolkit

The CUDA Toolkit does not need to be installed directly on the host when using the Docker build.

Verify GPU access:

```bash
nvidia-smi
```

Then verify Docker GPU access:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.1.2-runtime-ubuntu24.04 \
  nvidia-smi
```

## Build

From the repository root:

```bash
docker build \
  --progress=plain \
  -t ninfer-l4:sm89 .
```

## Model

Create the model directory:

```bash
mkdir -p models
```

The validated NVFP4 artifact is:

```text
neroued/Qwen3.8-27B-nvfp4-NInfer
qwen3_8_27b_nvfp4.ninfer
```

Validated revision:

```text
d6d0b3b61a38262e57217e64e7f44cf4ce98bda1
```

Expected SHA-256:

```text
bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
```

Verify the downloaded file:

```bash
echo \
"bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32  models/qwen3_8_27b_nvfp4.ninfer" \
| sha256sum -c -
```

Expected result:

```text
models/qwen3_8_27b_nvfp4.ninfer: OK
```

## Basic inference

```bash
docker run --rm \
  --gpus all \
  -v "$PWD/models:/workspace/models:ro" \
  ninfer-l4:sm89 \
  ninfer models/qwen3_8_27b_nvfp4.ninfer \
    --prompt "Write a C# method that calculates Fibonacci numbers." \
    --max-context 8192 \
    --kv-capacity 8192 \
    --max-new 256 \
    --kv-dtype int8 \
    --no-thinking
```

## MTP3

For interactive generation, enable MTP3:

```bash
docker run --rm \
  --gpus all \
  -v "$PWD/models:/workspace/models:ro" \
  ninfer-l4:sm89 \
  ninfer models/qwen3_8_27b_nvfp4.ninfer \
    --prompt "Write a C# implementation of an LRU cache with comments." \
    --max-context 8192 \
    --kv-capacity 8192 \
    --max-new 1000 \
    --kv-dtype int8 \
    --spec mtp \
    --draft-tokens 3 \
    --lm-head-draft \
    --no-thinking
```

A validated L4 run produced approximately:

```text
prefill speed        51.97 tok/s
decode speed         22.29 tok/s
overall              22.08 tok/s
MTP acceptance       75.90%
```

## Why NVFP4 is slower than expected

NVIDIA L4 is an Ada GPU.

It does not have the Blackwell NVFP4 Tensor Core execution path used by the original NInfer NVFP4 implementation.

This fork therefore executes NVFP4 weights through the A16/dequantization compatibility path.

Consequences:

- NVFP4 storage is supported.
- Native Blackwell FP4 compute performance is not available.
- NVFP4 is currently slower than the groupwise-int Qwen3.8 artifact on L4.

For production or interactive coding use, prefer the groupwise-int artifact unless testing NVFP4 compatibility is specifically required.

## L4 compatibility patches

### GDN cooperative routing

Inherited GDN routing tables can select cooperative kernels whose grid does not fit an L4.

The L4 fork changes route resolution so that an illegal high-parallelism route is demoted to the next legal route.

Conceptually:

```text
split8
  ↓ if residency is insufficient
split4
  ↓
split2
  ↓
unsplit
```

### NVFP4 A16 SwiGLU prefill

The inherited A16 NVFP4 SwiGLU path rejected workloads larger than 16 tokens.

The L4 fork:

- enables registered small-T schedules through 32 tokens;
- chunks larger prefill workloads into `<=32` token launches;
- uses the single-token decode kernel for a final one-token remainder.

For example:

```text
T=77
→ 32 + 32 + 13

T=65
→ 32 + 32 + 1
```

## Current validation scope

Validated:

- NVIDIA L4
- `sm_89`
- Linux
- Docker
- CUDA 13.1 container
- Qwen3.8-27B NVFP4
- INT8 group-64 KV
- 8K KV capacity
- MTP3
- text generation

Not yet qualified:

- long context NVFP4 profiles
- concurrency
- vision
- DFlash
- FP8 KV
- native installation outside Docker
- production stability
- optimized L4-specific CUDA schedules

## Performance work

The current changes are primarily compatibility fixes.

Future L4 optimization work should focus on:

- GDN launch geometry for 58 SMs;
- small-T GEMM/SwiGLU schedules;
- GQA decode wave sizing;
- register pressure and occupancy;
- Split-K selection;
- memory-bandwidth utilization.

All performance changes should be benchmarked against the groupwise-int baseline before being retained.
