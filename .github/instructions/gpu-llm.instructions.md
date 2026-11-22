---
description: GPU and LLM workload guidance for the local cluster
applyTo: "k8s/StarCoder2/**,GPU/**,llm/**,llama_factory/**,quantizer/**,CPU-AI-Orchestrator/**"
---

# GPU / LLM workloads

## GPU inventory
| Node | GPUs | Total VRAM | Notes |
|---|---|---|---|
| `kevin-ubuntu` | 2x RTX 5060 Ti (16 GB, Blackwell) | 32 GB | Fastest; host primary LLMs here. 16 CPU / ~32 GB RAM, so keep pod memory requests well under 32 Gi. |
| `rogue1` | 2x RTX 3060 (12 GB, Ampere) | 24 GB | Needs the `db-only` toleration. |
| `stormtrooper` | 1x RTX 2070 (8 GB, Turing) | 8 GB | Embedding models; no bf16 support, so use fp16/quantized models. 8 CPU / ~16 GB RAM, so keep pod memory limits modest (the current Ollama pod uses 8Gi). |
| `kevin-surface-book` | none | - | Do not schedule GPU workloads. Only 4 CPU / ~8 GB RAM. |

## Sizing
- A model plus its KV cache must fit in the **node's** total VRAM. Quantized weights + KV cache at the configured context + ~1-2 GB overhead per GPU.
- Splitting across 2 GPUs works in Ollama but is slower than a single GPU; use a single GPU when the model fits.
- A GPU is exclusive (time-slicing off): requesting `nvidia.com/gpu: N` consumes N whole GPUs, and a node's GPUs cannot be shared by two pods beyond their count.

## Ollama conventions (see `k8s\StarCoder2\deployment.yaml`)
- Image `ollama/ollama`, port `11434`, models mounted at `/root/.ollama` from a PVC.
- Multi-GPU: `OLLAMA_SCHED_SPREAD=1`; reduce memory with `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`.
- Context length: `OLLAMA_CONTEXT_LENGTH` (the existing rogue1-folder manifest uses `num_ctx`, which Ollama does not read as an env var).
- `NVIDIA_VISIBLE_DEVICES=all` plus `CUDA_VISIBLE_DEVICES` to choose ordering. On `kevin-ubuntu`, `"1,0"` is used to put the 5060 Ti first; verify indices with `nvidia-smi` before changing.
- Verify after rollout: `kubectl logs deploy/<name>` should show the expected GPUs and model offload, and `ollama ps` should show `100% GPU`.
