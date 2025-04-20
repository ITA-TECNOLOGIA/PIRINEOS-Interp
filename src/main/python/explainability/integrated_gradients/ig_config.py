from dataclasses import dataclass

@dataclass
class IGConfig:
    n_steps: int = 50
    internal_batch_size: int = 10
    return_convergence_delta: bool = False
    integration_method: str = "gausslegendre"   # Integral approximation method
    target: int = None
    multiply_by_inputs: bool = True
    prediction_method: str = "logprob"
    max_new_tokens: int = 512