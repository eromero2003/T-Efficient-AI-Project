import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def safe_read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_latency_vs_prefill(
    df: pd.DataFrame,
    title: str,
    out_path: str,
    device_filter: str | None = None,
    decode_tokens: int | None = None,
):
    if df.empty:
        print(f"No data for {title}")
        return

    if device_filter is not None:
        df = df[df["device"] == device_filter]
    if decode_tokens is not None:
        df = df[df["decode_tokens"] == decode_tokens]

    if df.empty:
        print(f"No matching rows for {title}")
        return

    plt.figure()
    for (model, quant), group in df.groupby(["model_name", "quantization"]):
        group_sorted = group.sort_values("prefill_tokens")
        plt.plot(
            group_sorted["prefill_tokens"],
            group_sorted["latency_seconds"],
            marker="o",
            label=f"{model}-{quant}",
        )
    plt.xlabel("Prefill tokens (approx)")
    plt.ylabel("Latency (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def plot_energy_vs_throughput(df: pd.DataFrame, title: str, out_path: str):
    if df.empty:
        print(f"No data for {title}")
        return
    df = df.dropna(subset=["energy_percent_per_token"])
    if df.empty:
        print(f"No energy data for {title}")
        return

    plt.figure()
    for (device, quant), group in df.groupby(["device", "quantization"]):
        plt.scatter(
            group["tokens_per_second"],
            group["energy_percent_per_token"],
            label=f"{device}-{quant}",
        )
    plt.xlabel("Tokens per second")
    plt.ylabel("Battery % per token (approx)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    baseline = safe_read(os.path.join(RESULTS_DIR, "baseline_results.csv"))
    phone = safe_read(os.path.join(RESULTS_DIR, "phone_results.csv"))
    laptop = safe_read(os.path.join(RESULTS_DIR, "laptop_results.csv"))

    plot_latency_vs_prefill(
        phone,
        "Galaxy S6 latency vs prefill tokens (decode=128)",
        os.path.join(RESULTS_DIR, "phone_latency_128.png"),
        device_filter="galaxy_s6",
        decode_tokens=128,
    )
    plot_latency_vs_prefill(
        laptop,
        "HP Envy latency vs prefill tokens (decode=128)",
        os.path.join(RESULTS_DIR, "laptop_latency_128.png"),
        device_filter="hp_envy",
        decode_tokens=128,
    )

    combined = pd.concat(
        [baseline, phone, laptop], ignore_index=True, sort=False
    )
    plot_energy_vs_throughput(
        combined,
        "Energy vs throughput (all devices)",
        os.path.join(RESULTS_DIR, "energy_vs_throughput.png"),
    )


if __name__ == "__main__":
    main()
