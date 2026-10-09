# NInfer for NVIDIA L4

Experimental NVIDIA L4 (`sm_89`) port of NInfer, based on `Don-Chad/ninfer-3090`, which in turn is derived from `Neroued/ninfer`.

The goal of this fork is to run and tune Qwen3.8-27B on a single NVIDIA L4 24 GB GPU.

## Current status

Validated configuration:

- NVIDIA L4 24 GB
- CUDA 13.1
- Linux
- Docker + NVIDIA Container Toolkit
- CUDA architecture `sm_89`
- Qwen3.8-27B
- NVFP4 weights through the A16 compatibility path
- INT8 group-64 KV cache
- MTP3 speculative decoding
- CLI inference

The NVFP4 artifact runs successfully on NVIDIA L4, but Ada does **not** provide Blackwell NVFP4 Tensor Core execution. NVFP4 weights are therefore executed through A16/dequantization compatibility kernels.

For performance-oriented L4 use, the groupwise-int artifact is currently recommended over NVFP4.

## L4-specific changes

This branch contains several changes required to run the upstream runtime on an NVIDIA L4:

1. Build CUDA kernels for `sm_89`.
2. Allow BF16 GDN gating to fall back to lower-parallelism cooperative routes when a route does not fit the GPU's SM residency.
3. Extend NVFP4 A16 SwiGLU small-T kernels through `T=32`.
4. Chunk larger NVFP4 A16 SwiGLU prefill workloads into `<=32` token launches.

The GDN fallback is important because the L4 has fewer SMs than the GPUs for which the inherited routing tables were tuned.

## Build

Requirements on the host:

- recent NVIDIA driver
- Docker
- NVIDIA Container Toolkit

Verify that Docker can access the GPU:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.1.2-runtime-ubuntu24.04 \
  nvidia-smi
```

Build:

```bash
docker build \
  --progress=plain \
  -t ninfer-l4:sm89 .
```

## Qwen3.8-27B NVFP4 model

The currently validated artifact is the container-v2 Qwen3.8-27B NVFP4 model.

Revision:

```text
d6d0b3b61a38262e57217e64e7f44cf4ce98bda1
```

SHA-256:

```text
bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32
```

Download it into `models/` using Hugging Face tooling.

Model artifacts are intentionally excluded from Git.

## Smoke test

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

## NVIDIA L4 measured result

Measured on one NVIDIA L4 with the configuration above:

| Metric | Result |
| --- | ---: |
| Prompt tokens | 25 |
| Generated tokens | 1000 |
| Prefill | 51.97 tok/s |
| Decode | 22.29 tok/s |
| Overall throughput | 22.08 tok/s |
| MTP draft window | 3 |
| MTP acceptance | 75.90% |
| MTP accepted tokens | 693 / 913 |
| GPU weights | 19.73 GiB |
| KV capacity | 8192 |
| KV payload | 280.63 MiB |
| Free VRAM after startup | 1.52 GiB |

These are early compatibility measurements, not optimized L4 performance numbers.

## NVFP4 vs groupwise-int

NVFP4 should not be assumed to be the fastest format on Ada.

In current testing on the same L4:

| Artifact | Decode with MTP3 |
| --- | ---: |
| Qwen3.8-27B groupwise-int | ~36.3 tok/s |
| Qwen3.8-27B NVFP4/A16 | ~22.3 tok/s |

The groupwise artifact is therefore currently the preferred option for interactive coding workloads on L4.

The NVFP4 path remains useful for compatibility work and kernel experimentation.

## Memory considerations

The NVFP4 + MTP3 configuration leaves relatively little VRAM available:

```text
GPU weights:             19.73 GiB
Free after weights:       2.11 GiB
Free after startup:       1.52 GiB
KV payload at 8K:       280.63 MiB
```

Only the 8K profile has been validated so far.

Larger context sizes should be tested explicitly before being advertised as supported. In particular, the current NVFP4 + MTP3 memory profile should not be assumed to support 64K context on a 24 GB L4.

## Upstream

This fork is based on:

```text
Don-Chad/ninfer-3090
    ↓
Neroued/ninfer
```

Performance numbers and hardware guarantees from the RTX 3090 or RTX 5090 projects do not automatically apply to the NVIDIA L4.

See `docs/l4-linux.md` for the L4-specific build and runtime guide.

The existing `docs/` directory contains the inherited NInfer CLI, HTTP API, model, benchmark, and architecture documentation.

## License

Apache License 2.0. See `LICENSE`.
