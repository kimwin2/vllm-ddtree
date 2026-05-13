#!/usr/bin/env python3
"""Summarization benchmark for the ddtree/dflash vLLM stack.

Loads a per-language summarization dataset (call_summary_short_<lang>.json)
and measures per-language:

  * accept_length  — derived from a /metrics snapshot delta around the
                     batch of requests for that language.
                     ``accept_length = 1 + Δ(accepted) / Δ(drafts)``
                     (1 = the bonus token target always contributes per
                     round, matching the SGLang ``spec_accept_length``
                     convention).
  * avg_tokens     — mean ``usage.completion_tokens`` across responses.
  * tps            — output tokens / wall-clock time for that language.

The script is server-mode only (talks to a running vllm server via the
OpenAI-compatible REST endpoints + the Prometheus /metrics endpoint),
so it works regardless of whether the server is dflash-baseline,
ddtree off, or ddtree on. To compare two configurations, run twice
against two different server configs and diff the resulting JSON.

Usage (mirrors the dflash.benchmark CLI conventions —
``--max-new-tokens``, ``--top-p``, ``--top-k``, ``--enable-thinking``,
``--timeout-s``):
    python tools/summarization_bench.py \\
        --base-url http://127.0.0.1:8000 \\
        --model google/gemma-4-E2B-it \\
        --data-dir /path/to/dataset_root \\
        --max-new-tokens 512 \\
        --temperature 0 --top-p 1.0 --top-k 1 \\
        --concurrency 1 \\
        --output-json /tmp/sum_bench_dflash.json

Dataset layout expected (per the team's existing summarization corpus):
    <data-dir>/call_summary_short_ko.json
    <data-dir>/call_summary_short_en.json
    ...
Each file is a JSON array of conversations; each conversation is a list
of >= 4 turn-dicts (``[{"content": ...}, ...]``) ordered as
[system, user_1, assistant_1, user_2]. The script sends the first four
turns and asks the server to continue with the assistant_2 response.

Stdlib + ``requests`` + ``numpy`` + ``tqdm`` only. No vllm or sglang
imports — keeps the script portable on the serving box.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests
from tqdm import tqdm


LANGS = [
    "ko", "en", "es", "fr", "de", "it", "pt", "zh",
    "ja", "pl", "hi", "th", "vi", "ar", "id", "ru",
]
TASK_PREFIX = "call_summary_short"

# Languages excluded from the secondary "avg minus" summary row (matches
# the team's existing reporting convention).
EXCLUDE_FROM_SUBSET = ("ar", "id", "ru")


def load_lang_data(base_dir: str, lang: str) -> list[dict[str, Any]]:
    """Load one language file. Returns ``[{'messages': [...]}, ...]``.

    Each conversation is expected to be a list of at least four
    turn-dicts; the first four are taken as
    ``[system, user_1, assistant_1, user_2]`` and re-wrapped into the
    OpenAI chat schema with explicit role tags.
    """
    filepath = os.path.join(base_dir, f"{TASK_PREFIX}_{lang}.json")
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict[str, Any]] = []
    for conv in data:
        if not isinstance(conv, list) or len(conv) < 4:
            continue
        try:
            messages = [
                {"role": "system",    "content": conv[0]["content"]},
                {"role": "user",      "content": conv[1]["content"]},
                {"role": "assistant", "content": conv[2]["content"]},
                {"role": "user",      "content": conv[3]["content"]},
            ]
        except (KeyError, TypeError):
            continue
        out.append({"messages": messages})
    return out


def get_spec_decode_metrics(base_url: str) -> dict[str, float]:
    """Snapshot the three counters we need from /metrics.

    Returns zeros if the endpoint is unreachable, so calling code can
    still produce meaningful timing-only stats on a non-spec-decode
    server (which would just yield accept_length = 1.0).
    """
    out = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0}
    try:
        r = requests.get(base_url + "/metrics", timeout=10)
        r.raise_for_status()
        text = r.text
    except Exception:
        return out
    for line in text.splitlines():
        # Lines look like:
        #   vllm:spec_decode_num_drafts_total{engine="0",model_name="..."} 79.0
        # We don't care about the labels; just the value after the last
        # whitespace on each matching line.
        if line.startswith("vllm:spec_decode_num_drafts_total{"):
            try:
                out["drafts"] = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            try:
                out["draft_tokens"] = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            try:
                out["accepted"] = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return out


def send_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    timeout_s: int,
) -> dict[str, Any]:
    """Match the JSON body shape used by ``dflash.benchmark._send_vllm``.

    The ``chat_template_kwargs.enable_thinking`` field is a vLLM
    extension that lets the chat template gate the thinking traces
    (Qwen3 / DeepSeek style). For Gemma-4 this is a no-op but we
    include it so the same request shape works across models.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    r = requests.post(
        base_url + "/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def run_language(
    base_url: str,
    model: str,
    samples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
    concurrency: int,
    timeout_s: int,
    desc: str = "samples",
) -> dict[str, Any]:
    """Run one language's samples. Returns its metrics dict.

    Brackets the run with /metrics snapshots so the accept length is
    truly per-language (rather than the cumulative server-lifetime
    value).
    """
    before = get_spec_decode_metrics(base_url)
    t0 = time.perf_counter()

    completion_tokens: list[int] = []
    errors: list[str] = []

    def send_one(messages: list[dict[str, str]]) -> int:
        try:
            resp = send_chat(
                base_url, model, messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                timeout_s=timeout_s,
            )
            return int(resp.get("usage", {}).get("completion_tokens", 0))
        except Exception as exc:  # network / 5xx / parse — count and skip
            errors.append(str(exc))
            return -1

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        futures = [pool.submit(send_one, s["messages"]) for s in samples]
        for fut in tqdm(
            as_completed(futures),
            total=len(samples),
            desc=desc,
            leave=False,
        ):
            n = fut.result()
            if n > 0:
                completion_tokens.append(n)

    elapsed = max(time.perf_counter() - t0, 1e-9)
    after = get_spec_decode_metrics(base_url)

    delta_drafts = max(after["drafts"] - before["drafts"], 0.0)
    delta_accepted = max(after["accepted"] - before["accepted"], 0.0)
    delta_draft_tokens = max(after["draft_tokens"] - before["draft_tokens"], 0.0)
    accept_per_draft = (
        delta_accepted / delta_drafts if delta_drafts > 0 else 0.0
    )
    accept_length = accept_per_draft + 1.0  # +1 bonus token per round

    avg_tokens = float(np.mean(completion_tokens)) if completion_tokens else 0.0
    tps = sum(completion_tokens) / elapsed

    return {
        "n_samples": len(samples),
        "n_ok": len(completion_tokens),
        "n_errors": len(errors),
        "elapsed_s": elapsed,
        "accept_length": accept_length,
        "accept_per_draft": accept_per_draft,
        "avg_completion_tokens": avg_tokens,
        "tps": tps,
        "delta_drafts": delta_drafts,
        "delta_draft_tokens": delta_draft_tokens,
        "delta_accepted": delta_accepted,
        "first_error": errors[0] if errors else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-language summarization benchmark on a vLLM server."
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--model",
        required=True,
        help="Model id served by vllm (e.g., google/gemma-4-E2B-it). "
             "If /v1/models reports a different id, that one is used.",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory holding call_summary_short_<lang>.json files.",
    )
    p.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help=f"Subset of languages to evaluate. Default: all 16 ({', '.join(LANGS)}).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Per-request output token cap. Default 512 (matches the "
             "SGLang summarization reference). dflash.benchmark uses "
             "2048 by default for non-summarization tasks.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling cutoff. Match dflash.benchmark default.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-K sampling cutoff. Match dflash.benchmark default (1).",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Forwarded to the chat template via "
             "``chat_template_kwargs.enable_thinking`` (Qwen3 / DeepSeek). "
             "Ignored by Gemma-4 templates.",
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument(
        "--max-samples-per-lang",
        type=int,
        default=None,
        help="Cap samples per language (default: no cap).",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help="Per-request HTTP timeout. Match dflash.benchmark default (3600).",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="If set, write per-language results to this JSON file.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup requests to send before measurement (helps "
             "settle CUDA graph + kv cache). Default 0 (skip).",
    )
    args = p.parse_args()

    base_url = args.base_url
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url

    langs = args.langs if args.langs else LANGS

    # Health check.
    try:
        requests.get(base_url + "/health", timeout=5).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"ERROR: server not reachable at {base_url}: {exc}")

    # Resolve the actually-served model id.
    served_id = args.model
    try:
        r = requests.get(base_url + "/v1/models", timeout=10)
        data = r.json()
        if data.get("data"):
            served_id = data["data"][0]["id"]
    except Exception:
        pass
    if served_id != args.model:
        print(f"NOTE: using served model id '{served_id}' (argv had '{args.model}')")

    # Load data.
    print(f"Loading data from {args.data_dir} ...")
    per_lang_samples: dict[str, list[dict[str, Any]]] = {}
    for lang in langs:
        s = load_lang_data(args.data_dir, lang)
        if args.max_samples_per_lang is not None:
            s = s[: args.max_samples_per_lang]
        if s:
            per_lang_samples[lang] = s
            print(f"  {lang}: {len(s)} samples")
        else:
            print(f"  {lang}: SKIP (file missing or empty)")

    if not per_lang_samples:
        raise SystemExit("No data loaded. Exiting.")

    # Optional warmup against the first available language.
    if args.warmup > 0:
        first_lang = next(iter(per_lang_samples))
        wm_samples = per_lang_samples[first_lang][: args.warmup]
        print(f"\nWarmup: {len(wm_samples)} requests on {first_lang} ...")
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
            list(pool.map(
                lambda s: send_chat(
                    base_url, served_id, s["messages"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    enable_thinking=args.enable_thinking,
                    timeout_s=args.timeout_s,
                ),
                wm_samples,
            ))

    # Run per language.
    total_samples = sum(len(s) for s in per_lang_samples.values())
    print(f"\nRunning ({total_samples} total samples) ...\n")
    results: dict[str, dict[str, Any]] = {}
    for lang in sorted(per_lang_samples):
        print(f"[{lang}] {len(per_lang_samples[lang])} samples ...")
        m = run_language(
            base_url, served_id, per_lang_samples[lang],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking=args.enable_thinking,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            desc=f"  {lang}",
        )
        results[lang] = m
        print(
            f"  → accept_length={m['accept_length']:.3f}  "
            f"avg_tokens={m['avg_completion_tokens']:.1f}  "
            f"tps={m['tps']:.2f}  "
            f"drafts={int(m['delta_drafts'])}  "
            f"accepted={int(m['delta_accepted'])}  "
            f"errors={m['n_errors']}"
        )

    # Print table.
    print()
    print("=" * 72)
    print("Summarization Benchmark Results")
    print(f"  base_url={base_url}  model={served_id}")
    print(
        f"  max_new_tokens={args.max_new_tokens}  "
        f"temperature={args.temperature}  "
        f"top_p={args.top_p}  top_k={args.top_k}  "
        f"enable_thinking={args.enable_thinking}"
    )
    print(
        f"  concurrency={args.concurrency}  "
        f"max_samples_per_lang={args.max_samples_per_lang}  "
        f"timeout_s={args.timeout_s}"
    )
    print("-" * 72)
    print(f"{'lang':<8}{'accept_len':>14}{'avg_tokens':>14}{'tps':>12}{'n_ok':>10}{'errors':>10}")

    sorted_langs = sorted(results)
    for lang in sorted_langs:
        m = results[lang]
        print(
            f"{lang:<8}"
            f"{m['accept_length']:>14.3f}"
            f"{m['avg_completion_tokens']:>14.1f}"
            f"{m['tps']:>12.2f}"
            f"{m['n_ok']:>10}"
            f"{m['n_errors']:>10}"
        )

    print("-" * 72)
    accs = [results[l]["accept_length"] for l in sorted_langs]
    lens = [results[l]["avg_completion_tokens"] for l in sorted_langs]
    if accs:
        print(
            f"{'avg':<8}"
            f"{np.mean(accs):>14.3f}"
            f"{np.mean(lens):>14.1f}"
        )
        wo = [l for l in sorted_langs if l not in EXCLUDE_FROM_SUBSET]
        if wo and len(wo) < len(sorted_langs):
            wo_accs = [results[l]["accept_length"] for l in wo]
            wo_lens = [results[l]["avg_completion_tokens"] for l in wo]
            excluded = ", ".join(
                l for l in EXCLUDE_FROM_SUBSET if l in sorted_langs
            )
            print(
                f"{'avg/sub':<8}"
                f"{np.mean(wo_accs):>14.3f}"
                f"{np.mean(wo_lens):>14.1f}  "
                f"(excluding {excluded})"
            )
    print("=" * 72)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "base_url": base_url,
                        "model_arg": args.model,
                        "served_model": served_id,
                        "data_dir": args.data_dir,
                        "langs": list(sorted_langs),
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "enable_thinking": args.enable_thinking,
                        "concurrency": args.concurrency,
                        "max_samples_per_lang": args.max_samples_per_lang,
                        "timeout_s": args.timeout_s,
                        "warmup": args.warmup,
                    },
                    "results_per_lang": results,
                    "summary": {
                        "avg_accept_length": float(np.mean(accs)) if accs else None,
                        "avg_completion_tokens": float(np.mean(lens)) if lens else None,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
