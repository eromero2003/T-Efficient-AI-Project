import argparse
import csv
import os
import time
from typing import List

from .config import (
    BASELINE_MODEL,
    BASELINE_PREFILL_TOKENS,
    BASELINE_DECODE_TOKENS,
    PREFILL_SWEEP,
    DECODE_SWEEP,
    DEVICE_PHONE,
    DEVICE_LAPTOP,
    PHONE_LAPTOP_MODELS,
)
from .food_dataset import load_meals_with_prompts
from .Model_interface import create_backend, ModelResult
from .metrics import get_system_snapshot, build_metric_row

RESULTS_DIR = "results"


def run_experiment(
    device_name: str,
    model_name: str,
    model_family: str,
    params_b: float,
    quantization: str,
    kv_quant: bool,
    prefill_tokens: int,
    decode_tokens: int,
    prompts: List[str],
    out_path: str,
    repeats: int = 1,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    backend = create_backend(model_name, quantization)

    fieldnames = None
    first_row = True

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = None

        for r in range(repeats):
            for i, prompt in enumerate(prompts):
                truncated_prompt = prompt[:prefill_tokens]

                sys_before = get_system_snapshot()
                start = time.perf_counter()
                result: ModelResult = backend.generate(
                    truncated_prompt,
                    max_tokens=decode_tokens,
                )
                end = time.perf_counter()
                sys_after = get_system_snapshot()

                latency = result.latency_seconds or (end - start)

                metric_row = build_metric_row(
                    device_name=device_name,
                    model_name=model_name,
                    quantization=quantization,
                    seq_len=prefill_tokens,
                    Model_result={
                        "tokens_generated": result.tokens_generated,
                        "latency_seconds": latency,
                        "prefill_seconds": result.prefill_seconds,
                        "decode_seconds": result.decode_seconds,
                    },
                    sys_before=sys_before,
                    sys_after=sys_after,
                )

                metric_row["model_family"] = model_family
                metric_row["params_b"] = params_b
                metric_row["kv_quant"] = int(kv_quant)
                metric_row["prefill_tokens"] = prefill_tokens
                metric_row["decode_tokens"] = decode_tokens
                metric_row["repeat_idx"] = r
                metric_row["prompt_idx"] = i
                metric_row["prompt_char_len"] = len(truncated_prompt)

                if first_row:
                    fieldnames = list(metric_row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    first_row = False
                else:
                    assert writer is not None

                writer.writerow(metric_row)
                f.flush()
                print(
                    f"[{device_name}] model={model_name}-{quantization} "
                    f"kvq={kv_quant} prefill={prefill_tokens} decode={decode_tokens} "
                    f"prompt={i+1}/{len(prompts)} repeat={r+1}/{repeats}"
                )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "phone_sweep", "laptop_sweep"],
        default="baseline",
    )
    parser.add_argument("--data_path", type=str, default="data/data.tsv")
    parser.add_argument("--items_path", type=str, default="data/items.tsv")
    parser.add_argument("--max_meals", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    prompts = load_meals_with_prompts(
       args.data_path,
       args.items_path,
       max_meals=args.max_meals,
    )

    if args.mode == "baseline":
        out_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
        run_experiment(
        device_name=DEVICE_LAPTOP.name,
        model_name=BASELINE_MODEL.name,
        model_family=BASELINE_MODEL.family,
        params_b=BASELINE_MODEL.params_b,
        quantization=BASELINE_MODEL.quantization,
        kv_quant=BASELINE_MODEL.kv_quant,
        prefill_tokens=BASELINE_PREFILL_TOKENS,
        decode_tokens=BASELINE_DECODE_TOKENS,
        prompts=prompts,
        out_path=out_path,
        repeats=args.repeats,
    )

    elif args.mode == "phone_sweep":
        out_path = os.path.join(RESULTS_DIR, "phone_results.csv")
        for model_cfg in PHONE_LAPTOP_MODELS:
            for prefill in PREFILL_SWEEP:
                for decode in DECODE_SWEEP:
                    run_experiment(
                    device_name=DEVICE_PHONE.name,
                    model_name=model_cfg.name,
                    model_family=model_cfg.family,
                    params_b=model_cfg.params_b,
                    quantization=model_cfg.quantization,
                    kv_quant=model_cfg.kv_quant,
                    prefill_tokens=prefill,
                    decode_tokens=decode,
                    prompts=prompts,
                    out_path=out_path,
                    repeats=args.repeats,
                )

    elif args.mode == "laptop_sweep":
        out_path = os.path.join(RESULTS_DIR, "laptop_results.csv")
        for model_cfg in PHONE_LAPTOP_MODELS:
            for prefill in PREFILL_SWEEP:
                for decode in DECODE_SWEEP:
                    run_experiment(
                    device_name=DEVICE_LAPTOP.name,
                    model_name=model_cfg.name,
                    model_family=model_cfg.family,
                    params_b=model_cfg.params_b,
                    quantization=model_cfg.quantization,
                    kv_quant=model_cfg.kv_quant,
                    prefill_tokens=prefill,
                    decode_tokens=decode,
                    prompts=prompts,
                    out_path=out_path,
                    repeats=args.repeats,
                )

if __name__ == "__main__":
    main()
