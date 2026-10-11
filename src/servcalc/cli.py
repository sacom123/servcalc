"""Command-line interface: `servcalc MODEL --gpu NAME ...` and `servcalc gpus`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from vramcalc import UnsupportedModelError
from vramcalc import gpus as _vg

from . import profiles
from .estimate import estimate
from .gpus import SPECS
from .modes import DoesNotFitError
from .workload import Workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="servcalc", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    est = sub.add_parser("estimate", help="predict serving capacity (default when MODEL is given)")
    est.add_argument("model", help="config.json path, model directory, or Hub id")
    est.add_argument("--gpu", required=True, help="GPU name from `servcalc gpus`")
    est.add_argument("--engine", default="vllm", choices=sorted(profiles.BUILTIN))
    est.add_argument("--prompt", type=int, default=1024, help="prompt tokens per request")
    est.add_argument("--gen", type=int, default=512, help="generated tokens per request")
    est.add_argument("--concurrency", default="1,2,4,8,16,32,64,128,256",
                     help="comma-separated batch sizes to evaluate")
    est.add_argument("--quant", help="weight format (awq, gptq, fp8, gguf_q4_k_m, ...)")
    est.add_argument("--kv-bits", type=int, help="KV cache bits (8 or 16)")
    est.add_argument("--max-seqs", type=int, help="engine sequence cap (ollama: num_parallel)")
    est.add_argument("--num-ctx", type=int, help="ollama num_ctx")
    est.add_argument("--token-budget", type=int, help="scheduler tokens per step")
    est.add_argument("--no-chunked", action="store_true", help="vllm: disable chunked prefill")
    est.add_argument("--slo-itl", type=float, metavar="MS",
                     help="report max users whose mean ITL is within MS")
    est.add_argument("--json", action="store_true")
    sub.add_parser("gpus", help="list GPUs with bandwidth and bf16 TFLOPs")
    return parser


def _profile(a: argparse.Namespace):
    touched = any(v is not None for v in (a.quant, a.kv_bits, a.max_seqs, a.num_ctx,
                                          a.token_budget)) or a.no_chunked
    if not touched:
        return a.engine   # bare name: quant is inferred from the config
    quant = {"quant": a.quant} if a.quant is not None else {}
    if a.engine == "vllm":
        p = profiles.vllm(chunked=not a.no_chunked, **quant)
    elif a.engine == "ollama":
        kwargs = dict(quant)
        if a.max_seqs is not None:
            kwargs["num_parallel"] = a.max_seqs
        if a.num_ctx is not None:
            kwargs["num_ctx"] = a.num_ctx
        p = profiles.ollama(**kwargs)
    else:
        p = profiles.hf(**quant)
    overrides = {}
    if a.kv_bits is not None:
        overrides["kv_bits"] = a.kv_bits
    if a.max_seqs is not None and a.engine != "ollama":
        overrides["max_seqs"] = a.max_seqs
    if a.token_budget is not None:
        overrides["token_budget"] = a.token_budget
    return dataclasses.replace(p, **overrides) if overrides else p


def cmd_estimate(a: argparse.Namespace) -> int:
    conc = tuple(int(x) for x in a.concurrency.split(","))
    e = estimate(a.model, a.gpu, workload=Workload(a.prompt, a.gen, conc), profile=_profile(a))
    if a.json:
        d = e.to_dict()
        if a.slo_itl is not None:
            d["max_users"] = e.max_users(a.slo_itl)
        print(json.dumps(d, indent=2))
        return 0
    print(e.table())
    if a.slo_itl is not None:
        print(f"\nmax users with mean ITL <= {a.slo_itl:g} ms: {e.max_users(a.slo_itl)}")
    for w in e.warnings:
        print(f"warning: {w}")
    return 0


def cmd_gpus(_: argparse.Namespace) -> int:
    print(f"{'GPU':<20} {'VRAM GiB':>9} {'GB/s':>7} {'bf16 TFLOPs':>12}")
    for name, vram in _vg.GPUS.items():
        bw, tf = SPECS[name]
        print(f"{name:<20} {vram / 2**30:>9.0f} {bw:>7.0f} {tf:>12.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("gpus", "estimate") and not argv[0].startswith("-"):
        argv.insert(0, "estimate")
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.command is None:
        parser.print_help()
        return 2
    try:
        return {"estimate": cmd_estimate, "gpus": cmd_gpus}[a.command](a)
    except DoesNotFitError as exc:
        print(f"servcalc: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, FileNotFoundError, ImportError, UnsupportedModelError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        print(f"servcalc: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
