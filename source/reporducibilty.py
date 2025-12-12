import argparse
import os

from .benchmark import run_experiment
from .config import (
    BASELINE_MODEL,
    BASELINE_PREFILL_TOKENS,
    BASELINE_DECODE_TOKENS,
)
from .food_dataset import load_meals_with_prompts

RESULTS_PATH = "results/reproducibility.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_name", required=True)
    parser.add_argument("--label", required=True, help="e.g. cool, warm, plugged_in")
    parser.add_argument("--data_path", default="data/data.tsv")
    parser.add_argument("--items_path", default="data/items.tsv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_meals", type=int, default=10)
    args = parser.parse_args()

    meals = load_meals_with_prompts(
        data_path=args.data_path,
        items_path=args.items_path,
        max_meals=args.max_meals,
    )
    prompts = meals

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    run_experiment(
        device_name=f"{args.device_name}_{args.label}",
        model_name=BASELINE_MODEL.name,
        model_family=BASELINE_MODEL.family,
        params_b=BASELINE_MODEL.params_b,
        quantization=BASELINE_MODEL.quantization,
        kv_quant=BASELINE_MODEL.kv_quant,
        prefill_tokens=BASELINE_PREFILL_TOKENS,
        decode_tokens=BASELINE_DECODE_TOKENS,
        prompts=prompts,
        out_path=RESULTS_PATH,
       repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
