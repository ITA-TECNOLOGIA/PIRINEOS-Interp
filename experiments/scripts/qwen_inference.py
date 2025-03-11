# Load model and sample prompt
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

from huggingface_hub import login

import os

access_token_write = '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'
login(
  token=access_token_write,
  add_to_git_credential=True
)

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-3B"
device = "cuda" # the device to load the model onto

# Quantization config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True, #This is required to load Qwen models.
    quantization_config=bnb_config # Apply 4-bit quantization
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

input_text = "What is the derivative of sin(x)?"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)

# Generate a completion with a specified number of new tokens
generated_ids = model.generate(input_ids, max_new_tokens=50)

# Decode the full generated sequence (prompt + completion)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)



