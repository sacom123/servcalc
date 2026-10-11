# Changelog

## 0.1.0 (2026-10-11)

- Roofline estimate of concurrency, tok/s, TTFT and ITL per batch size for dense decoder-only models on one GPU
- Engine profiles: vllm (chunked prefill, paged KV), ollama (GGUF, preallocated context), hf (static generate)
- Ranges from prior efficiency curves; no measured calibration yet
- CLI: `servcalc MODEL --gpu NAME`, `servcalc gpus`
