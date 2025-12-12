from dataclasses import dataclass
from typing import Literal, List
from pathlib import Path
from typing import List
import pandas as pd

QuantLevel = Literal["int8", "int4"]
DeviceKind = Literal["phone", "laptop"]


@dataclass
class DeviceConfig:
    name: str          
    kind: DeviceKind  
    description: str  


@dataclass
class ModelConfig:
    name: str          
    family: str       
    params_b: float    
    quantization: QuantLevel
    kv_quant: bool    


DEVICE_PHONE = DeviceConfig(
    name="galaxy_s6",
    kind="phone",
    description="Samsung Galaxy S6",
)

DEVICE_LAPTOP = DeviceConfig(
    name="hp_envy",
    kind="laptop",
    description="HP Envy laptop",
)

PHONE_LAPTOP_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="phi-3-mini-3.8b",
        family="phi-3",
        params_b=3.8,
        quantization="int8",
        kv_quant=False,
    ),
    ModelConfig(
        name="phi-3-mini-3.8b",
        family="phi-3",
        params_b=3.8,
        quantization="int4",
        kv_quant=True,
    ),
    ModelConfig(
        name="tinyllama-1.1b",
        family="tinyllama",
        params_b=1.1,
        quantization="int4",
        kv_quant=True,
    ),
]

BASELINE_MODEL = ModelConfig(
    name="phi-3-mini-3.8b",
    family="phi-3",
    params_b=3.8,
    quantization="int8",
    kv_quant=False,
)

BASELINE_PREFILL_TOKENS = 1024
BASELINE_DECODE_TOKENS = 128

PREFILL_SWEEP = [256, 1024, 2048]
DECODE_SWEEP = [128, 512]

def load_meals_with_prompts(
    data_path: str,
    items_path: str,
    max_meals: int | None = None,
) -> List[str]:
    data_path = Path(data_path)
    items_path = Path(items_path)


    df_data = pd.read_csv(data_path, sep="\t")
    df_items = pd.read_csv(items_path, sep="\t")

    id_to_item = dict(zip(df_items["id"], df_items["item"]))

    prompts: List[str] = []

    for _, row in df_data.iterrows():
        food_ids_raw = str(row["food_ids"]).split(",")

        ingredient_names: list[str] = []
        for fid in food_ids_raw:
            fid = fid.strip()
            if not fid:
                continue

            try:
                key = int(fid)
            except ValueError:
                key = fid

            item_taxonomy = id_to_item.get(key)
            if not item_taxonomy:
                continue

            for segment in str(item_taxonomy).split(","):
                parts = segment.split("__")
                leaf = parts[-1] if parts else segment
                ingredient_names.append(leaf)

        if not ingredient_names:
            continue

        seen = set()
        ordered_names: list[str] = []
        for name in ingredient_names:
            name_clean = name.replace("_", " ")
            if name_clean not in seen:
                seen.add(name_clean)
                ordered_names.append(name_clean)

        meal_list = ", ".join(ordered_names)
        date = row["date"]
        seq = row["meal_sequence"]
        user = row["user_id"]

        prompt = (
            f"Food entry for user {user} on {date}, meal {seq}. "
            f"The meal contains: {meal_list}. "
            f"Give summary of meal and break down its macros."
        )

        prompts.append(prompt)

        if max_meals is not None and len(prompts) >= max_meals:
            break

    return prompts
