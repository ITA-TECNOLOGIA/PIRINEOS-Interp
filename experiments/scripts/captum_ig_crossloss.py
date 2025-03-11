from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from huggingface_hub import login
from captum.attr import IntegratedGradients



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

# Prepare the input
input_text = "What is the derivative of sin(x)?"
#input_text = "((125 + 378 * 4 - 56 / 7) * (3 / 2) + 1024 - (45 * 6 - 98 / 2) + (789 / 3 + 256 * 5) - 678)"

inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)


# Define a forward function returning a scalar for attribution
def forward_func(input_embeds, attention_mask, max_new_tokens = 50, eos_token_id = tokenizer.eos_token_id):
    # In order not to overwrite
    current_input_embeds = input_embeds.clone()
    current_attention_mask = attention_mask.clone()
    
    # To store all the resulting log probs
    batch_size = current_input_embeds.size(0)
    sum_log_probs = torch.zeros(batch_size, device=current_input_embeds.device, dtype=torch.float32)

    # To embed when we get the new token iteratively
    token_embed_layer = model.get_input_embeddings()
    
    for _ in range(max_new_tokens):
        # In this loop we iterate over the new tokens, updating the input adding new token each time   
        # Importante trackear el shape porque el IG mete los steps como batch size
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        
        # get the outputs we are interested in
        last_logits = outputs.logits[:, -1, :]          # shape: [batch_size, vocab_size], get the output for the last input token
        log_probs = F.log_softmax(last_logits, dim=-1)  # shape: [batch_size, vocab_size]
        next_tokens = torch.argmax(log_probs, dim=-1)   # shape: [batch_size]
        next_tokens_log_prob = log_probs[torch.arange(batch_size), next_tokens]  # shape: [batch_size]
        sum_log_probs += next_tokens_log_prob   # lo agregamos al tensor de todo zeros, sumandolo al resto de su batch
        
        # Checkeamos si ha acabado en todas las secuencias del batch
        done_mask = (next_tokens == eos_token_id)  # shape: [batch_size]
        if done_mask.all():
            break
            
        # Ahora creamos el nuevo input iterativamente
        next_tokens_embeds = token_embed_layer(next_tokens).unsqueeze(1)
        current_input_embeds = torch.cat([current_input_embeds, next_tokens_embeds], dim=1)
        next_tokens_mask = torch.ones(
            (batch_size, 1), 
            dtype=current_attention_mask.dtype, 
            device=current_attention_mask.device
        )
        current_attention_mask = torch.cat([current_attention_mask, next_tokens_mask], dim=1)
        
    return sum_log_probs.unsqueeze(-1)  # shape [batch_size, 1]


# 5) Prepare the Integrated Gradients object
ig = IntegratedGradients(forward_func)


"""
# to compare with an specific baseline
# Prepare the input
baseline_text = "What is the derivative of cos(x)?"
baseline = tokenizer(baseline_text, return_tensors="pt")
baseline_ids = baseline["input_ids"].to(model.device)

# Convert the input_ids to embeddings so Captum can do gradient-based methods
with torch.no_grad():
    token_embed_layer = model.get_input_embeddings() 
    # Actual input embeddings
    input_embeds = token_embed_layer(input_ids)
    baseline_embed_layer = token_embed_layer(baseline_ids)
"""
    
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
# the attributions for the first example
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for token_str, score in zip(tokens, token_level_attributions[0].tolist()):
    print(f"{token_str} => {score:.4f}")
