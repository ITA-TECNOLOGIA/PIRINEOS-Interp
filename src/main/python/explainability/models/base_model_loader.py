from abc import ABC, abstractmethod

from explainability.models.llm_config import LLMConfig

class BaseModelLoader(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.hf_login(None)

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load the model, must be overridden.
        """
        pass
    
    @abstractmethod
    def hf_login(self):
        """
        Abstract method to handle Hugging Face login, must be overridden.
        """
        pass
    
    def prompt_generate(self, input_text, max_new_tokens = 50):
        """
        Generate a completion for the given input text using the model.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        # Generate a completion with a specified number of new tokens
        generated_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)

        # Decode the full generated sequence (prompt + completion)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text