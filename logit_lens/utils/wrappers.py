import torch

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, norm):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        
        # Wrap attention module
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        
        # Store intermediate activations
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        
    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        output = self.block(x, past_key_value=past_key_value, attention_mask=attention_mask, 
                          position_ids=position_ids, **kwargs)
        
        # Get hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Get attention output and compute intermediate activations
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.lm_head(self.norm(attn_output))
        
        # Add residual connection
        attn_output += x
        self.intermediate_res_unembedded = self.lm_head(self.norm(attn_output))
        
        # MLP output
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.lm_head(self.norm(mlp_output))
        
        # Block output
        self.block_output_unembedded = self.lm_head(self.norm(hidden_states))

        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def get_attn_activations(self):
        return self.block.self_attn.activations