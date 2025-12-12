from dataclasses import dataclass
from typing import Literal, List

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
    name="galaxy_s6_sim",
    kind="phone",
    description="Galaxy S6 simulation",
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
