from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from captum.attr import IntegratedGradients

import os

access_token_write = os.getenv("HF_TOKEN")
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

# Prepare the input
input_text = "What is the derivative of sin(x)?"

inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)


# Define a forward function returning a scalar for attribution
def forward_func(input_embeds, attention_mask):
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    # outputs.logits shape = [batch_size, seq_len, vocab_size]

    # Get the logits for the final token in each sequence
    last_token_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
    # El batch size es el step size del ig, porque le mete como batch

    # Identify the token index with the maximum logit (the "best" prediction)
    best_token_ids = torch.argmax(last_token_logits, dim=-1)  # shape: [batch_size]

    # Gather the corresponding max logits; shape will be [batch_size, 1]
    best_logits = last_token_logits.gather(
        1, best_token_ids.unsqueeze(-1)
    )

    # Return [batch_size, 1] so each example has its own single scalar
    return best_logits

# 5) Prepare the Integrated Gradients object
ig = IntegratedGradients(forward_func)

max_new_tokens = 50

for step in range(max_new_tokens):
    
    # Convert the input_ids to embeddings so Captum can do gradient-based methods
    with torch.no_grad():
        token_embed_layer = model.get_input_embeddings() 
        # Actual input embeddings
        input_embeds = token_embed_layer(input_ids)
        baseline_embed_layer = torch.zeros_like(token_embed_layer(input_ids))

    # Compute attributions
    attributions = ig.attribute(
        input_embeds,
        baselines=baseline_embed_layer,
        additional_forward_args=(attention_mask,),
        # e.g. "gausslegendre" or "riemann_trapezoid", etc.
        method="gausslegendre",
        n_steps=50,  # step count for interpolation
    )

    # get the token-level attributions by summing the attributions across the last dimension
    token_level_attributions = attributions.sum(dim=-1)  # shape: [batch_size, seq_len]
    print(f"Step {step}: Current input =>", tokenizer.decode(input_ids[0]))
    # the attributions for the first example
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    for token_str, score in zip(tokens, token_level_attributions[0].tolist()):
        print(f"{token_str} => {score:.4f}")
    print("----")
        
    # Now find actual output to cat with the previous input and generate the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        next_token_id = torch.argmax(last_logits, dim=-1).unsqueeze(-1)  # [batch_size, 1]

    # Append the new token to the input sequence
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    # update attention mask accordingly
    new_mask = torch.ones_like(next_token_id, dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
    
    if next_token_id.item() == tokenizer.eos_token_id:
        break