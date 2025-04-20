from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str
    device_map: str = "auto"
    torch_dtype: str = "auto"
    quant_params: dict = None