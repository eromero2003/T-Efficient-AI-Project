from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional, Tuple

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    Llama = None  
    _HAS_LLAMA_CPP = False


@dataclass
class ModelResult:
    prompt: str
    output: str
    tokens_generated: int
    latency_seconds: float
    prefill_seconds: Optional[float]
    decode_seconds: Optional[float]
    metadata: Dict[str, Any]


class ModelBackend(Protocol):
    def generate(self, prompt: str, max_tokens: int = 128) -> ModelResult:
        ...


class SimulatedBackend:
    def __init__(self, model_name: str, quantization: str):
        self.model_name = model_name
        self.quantization = quantization

    def generate(self, prompt: str, max_tokens: int = 128) -> ModelResult:
        prefill_start = time.perf_counter()
        time.sleep(0.03)
        prefill_end = time.perf_counter()

        decode_start = prefill_end
        time.sleep(0.07)
        decode_end = time.perf_counter()

        output = (
            "Simulated output with summary of your meal and a recomended recipe."
        )
        tokens_generated = len(output.split())

        return ModelResult(
            prompt=prompt,
            output=output,
            tokens_generated=tokens_generated,
            latency_seconds=decode_end - prefill_start,
            prefill_seconds=prefill_end - prefill_start,
            decode_seconds=decode_end - decode_start,
            metadata={
                "backend": "Simulated",
                "model_name": self.model_name,
                "quantization": self.quantization,
                "max_tokens": max_tokens,
            },
        )


class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        model_name: str = "",
        quantization: str = "",
    ):
        if not _HAS_LLAMA_CPP:
            raise RuntimeError(
                "Install llama-cpp-python please!"
            )

        self.model_path = model_path
        self.model_name = model_name or model_path
        self.quantization = quantization

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers

    def generate(self, prompt: str, max_tokens: int = 128) -> ModelResult:
        start = time.perf_counter()

        text_chunks = []
        first_token_time: Optional[float] = None

        for chunk in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            echo=False,
            stream=True,
        ):
            now = time.perf_counter()
            token_text = chunk["choices"][0]["text"]

            if first_token_time is None:
                first_token_time = now

            text_chunks.append(token_text)

        end = time.perf_counter()

        output_text = "".join(text_chunks)
        latency = end - start


        if first_token_time is None:

            prefill_seconds = latency
            decode_seconds = 0.0
        else:
            prefill_seconds = first_token_time - start
            decode_seconds = end - first_token_time

        tokens_generated = len(output_text.split())

        return ModelResult(
            prompt=prompt,
            output=output_text,
            tokens_generated=tokens_generated,
            latency_seconds=latency,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            metadata={
                "backend": "llama_cpp",
                "model_name": self.model_name,
                "quantization": self.quantization,
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_gpu_layers": self.n_gpu_layers,
            },
        )


_MODEL_REGISTRY: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("phi-3-mini-3.8b", "int8"): {
        "model_path": "models/phi-3-mini-3.8b-int8.gguf",
        "n_ctx": 4096,
        "n_threads": 6,
        "n_gpu_layers": 0,
    },
    ("phi-3-mini-3.8b", "int4"): {
        "model_path": "models/phi-3-mini-3.8b-int4.gguf",
        "n_ctx": 4096,
        "n_threads": 6,
        "n_gpu_layers": 0,
    },
    ("tinyllama-1.1b", "int4"): {
        "model_path": "models/tinyllama-1.1b-int4.gguf",
        "n_ctx": 4096,
        "n_threads": 6,
        "n_gpu_layers": 0,
    },
}


def create_backend(model_name: str, quantization: str) -> ModelBackend:
    key = (model_name, quantization)
    cfg = _MODEL_REGISTRY.get(key)

    if cfg is not None and _HAS_LLAMA_CPP:
        print(f"[Model_interface] llama.cpp backend in use for {model_name}-{quantization}")
        return LlamaCppBackend(
            model_path=cfg["model_path"],
            n_ctx=cfg.get("n_ctx", 4096),
            n_threads=cfg.get("n_threads", 4),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            model_name=model_name,
            quantization=quantization,
        )

    print(
        f"[Model_interface] SimulatedBackend in use for {model_name}-{quantization} "
        f"(Install llama-cpp-python please!)."
    )
    return SimulatedBackend(model_name=model_name, quantization=quantization)
