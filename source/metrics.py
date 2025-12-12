from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import psutil


@dataclass
class SystemSnapshot:
    timestamp: float
    cpu_percent: float
    ram_bytes: int
    battery_percent: Optional[float]
    battery_power_plugged: Optional[bool]
    temperature_celsius: Optional[float]


def get_system_snapshot() -> SystemSnapshot:
    process = psutil.Process()
    cpu = psutil.cpu_percent(interval=None)
    mem = process.memory_info().rss

    try:
        battery = psutil.sensors_battery()
    except Exception:
        battery = None

    battery_percent = battery.percent if battery else None
    battery_plugged = battery.power_plugged if battery else None

    temp_c = None
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            first_sensor = next(iter(temps.values()))
            if first_sensor:
                temp_c = first_sensor[0].current
    except Exception:
        temp_c = None

    return SystemSnapshot(
        timestamp=time.time(),
        cpu_percent=cpu,
        ram_bytes=mem,
        battery_percent=battery_percent,
        battery_power_plugged=battery_plugged,
        temperature_celsius=temp_c,
    )


def build_metric_row(
    device_name: str,
    model_name: str,
    quantization: str,
    seq_len: int,
    Model_result: Dict[str, Any],
    sys_before: SystemSnapshot,
    sys_after: SystemSnapshot,
) -> Dict[str, Any]:
    tokens_generated = Model_result["tokens_generated"]
    latency = Model_result["latency_seconds"]
    prefill = Model_result.get("prefill_seconds")
    decode = Model_result.get("decode_seconds")

    tokens_per_sec = tokens_generated / latency if latency > 0 else 0.0

    row: Dict[str, Any] = {
        "device": device_name,
        "model_name": model_name,
        "quantization": quantization,
        "seq_len": seq_len,
        "tokens_generated": tokens_generated,
        "latency_seconds": latency,
        "tokens_per_second": tokens_per_sec,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
    }

    before = asdict(sys_before)
    after = asdict(sys_after)

    for prefix, snap in [("before_", before), ("after_", after)]:
        for k, v in snap.items():
            row[prefix + k] = v

    if (before.get("battery_percent") is not None
            and after.get("battery_percent") is not None):
        delta_batt = before["battery_percent"] - after["battery_percent"]
        row["delta_battery_percent"] = delta_batt
        row["energy_percent_per_token"] = (
            delta_batt / tokens_generated if tokens_generated > 0 else None
        )
    else:
        row["delta_battery_percent"] = None
        row["energy_percent_per_token"] = None

    return row
