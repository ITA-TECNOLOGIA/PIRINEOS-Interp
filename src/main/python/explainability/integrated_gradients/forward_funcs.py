import torch
import torch.nn.functional as F


    


def logprob_generate_forward_func(input_embeds, attention_mask, model, tokenizer, max_new_tokens = 10):
    """
    Usage: lambda a,b,c,d: logprob_forward_func(a,b,c,d, max_new_tokens=10)
    """
    # In order not to overwrite
    current_input_embeds = input_embeds.clone()
    current_attention_mask = attention_mask.clone()
    
    # To store all the resulting log probs
    batch_size = current_input_embeds.size(0)
    sum_log_probs = torch.zeros(batch_size, device=current_input_embeds.device, dtype=torch.float32)
    token_counts = torch.zeros(batch_size, device=current_input_embeds.device, dtype=torch.float32)

    # To embed when we get the new token iteratively
    token_embed_layer = model.get_input_embeddings()
    for _ in range(max_new_tokens):
        # In this loop we iterate over the new tokens, updating the input adding new token each time   
        # Importante trackear el shape porque el IG mete los steps como batch size
        outputs = model(inputs_embeds=current_input_embeds, attention_mask=current_attention_mask)
        
        # get the outputs we are interested in
        last_logits = outputs.logits[:, -1, :]          # shape: [batch_size, vocab_size], get the output for the last input token
        log_probs = F.log_softmax(last_logits, dim=-1)  # shape: [batch_size, vocab_size]
        next_tokens = torch.argmax(log_probs, dim=-1)   # shape: [batch_size]
        
        next_tokens_log_prob = log_probs[torch.arange(batch_size), next_tokens]  # shape: [batch_size]
        not_done_mask = (next_tokens != tokenizer.eos_token_id)   # shape: [batch_size]
        
        sum_log_probs += next_tokens_log_prob*not_done_mask   # lo agregamos al tensor de todo zeros, sumandolo al resto de su batch. Pero solo si no ha acabado la generación
        token_counts += not_done_mask.float()
        
        # Checkeamos si ha acabado en todas las secuencias del batch
        if (~not_done_mask).all():
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
        
    avg_log_probs = sum_log_probs / token_counts.clamp(min=1.0)
    return avg_log_probs.unsqueeze(-1)  # shape [batch_size, 1]

def entropy_generate_forward_func(input_embeds, attention_mask, model, tokenizer, max_new_tokens = 10):
    """
    Usage: lambda a,b,c,d: logprob_forward_func(a,b,c,d, max_new_tokens=10)
    For now, average over generated tokens is computed
    """
    # In order not to overwrite
    current_input_embeds = input_embeds.clone()
    current_attention_mask = attention_mask.clone()
    
    # To store all the resulting log probs
    batch_size = current_input_embeds.size(0)
    sum_log_probs = torch.zeros(batch_size, device=current_input_embeds.device, dtype=torch.float32)
    token_counts = torch.zeros(batch_size, device=current_input_embeds.device, dtype=torch.float32)

    # To embed when we get the new token iteratively
    token_embed_layer = model.get_input_embeddings()
    for _ in range(max_new_tokens):
        # In this loop we iterate over the new tokens, updating the input adding new token each time   
        # Importante trackear el shape porque el IG mete los steps como batch size
        outputs = model(inputs_embeds=current_input_embeds, attention_mask=current_attention_mask)
        
        # get the outputs we are interested in
        last_logits = outputs.logits[:, -1, :]          # shape: [batch_size, vocab_size], get the output for the last input token
        log_probs = F.log_softmax(last_logits, dim=-1)  # shape: [batch_size, vocab_size]
        probs = torch.exp(log_probs)                      # shape: [batch_size, vocab_size]
        next_tokens = torch.argmax(log_probs, dim=-1)   # shape: [batch_size]
        
        entropy = -(probs * log_probs).sum(dim=-1)
        not_done_mask = (next_tokens != tokenizer.eos_token_id)   # shape: [batch_size]
        
        total_entropy += entropy * not_done_mask
        token_counts += not_done_mask.float()
        
        # Checkeamos si ha acabado en todas las secuencias del batch
        if (~not_done_mask).all():
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
        
    avg_entropy = total_entropy / token_counts.clamp(min=1.0)
    return avg_entropy.unsqueeze(-1)