import getpass

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from explainability.models.base_model_loader import BaseModelLoader

class DefaultHFModelLoader(BaseModelLoader):
    def load_model(self):
        # Pseudocode for normal loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True, #This is required to load Qwen models.
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self.model, self.tokenizer
    
    def hf_login(self, access_token):
        """
        Login to Hugging Face using the provided access token.
        """
        if access_token:
            # Use the provided access token to login
            login(token=access_token, add_to_git_credential=True)
        else:
            # If no access token is provided, prompt for token
            access_token = getpass.getpass("Enter your Hugging Face access token: ")
            login(token=access_token, add_to_git_credential=True)

class BnBQuantizedModelLoader(BaseModelLoader):
    def load_model(self):
        # Quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        # Load model from huggingface
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True, #This is required to load Qwen models.
            quantization_config=quantization_config # Apply 4-bit quantization
        )
        # Tokenizer too
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return model, tokenizer
    
    def hf_login(self, access_token):
        """
        Login to Hugging Face using the provided access token.
        """
        if access_token:
            # Use the provided access token to login
            login(token=access_token, add_to_git_credential=True)
        else:
            # If no access token is provided, prompt for token
            access_token = getpass.getpass("Enter your Hugging Face access token: ")
            login(token=access_token, add_to_git_credential=True)

