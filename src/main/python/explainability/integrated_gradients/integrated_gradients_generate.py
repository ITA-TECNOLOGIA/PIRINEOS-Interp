import torch
from captum.attr import IntegratedGradients

class IGGenerate:
    def __init__(self, model, tokenizer, forward_func = None):
        """
        Initialize the Integrated Gradients for sequence generation.
        
        Args:
            model: The model to be used for generating sequences.
            tokenizer: The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # If forward_func is provided, load IntegratedGradients with it
        if forward_func:
            self.ig = IntegratedGradients(forward_func)
        else:
            self.ig = None
            
    def load_ig(self, forward_func):
        """
        Load the Integrated Gradients with the provided forward function.
        Can be used to reuse the model for different forward functions.
        
        Args:
            forward_func: The forward function to be used for attribution.
        """
        self.ig = IntegratedGradients(forward_func)
        
    def compute_attributions(self, input_text, n_steps=50, integral_approximation_method='gausslegendre'):
        # Get input ids and attention mask for that input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Get input embeddings from ids and baseline for inputs trajectory
        with torch.no_grad():
            token_embed_layer = self.model.get_input_embeddings()
            input_embeds = token_embed_layer(input_ids)
            baseline_embeds = torch.zeros_like(input_embeds)
        
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=n_steps,
            internal_batch_size = 10,
            method=integral_approximation_method,
            additional_forward_args=(attention_mask, self.model, self.tokenizer, ),
        )
        
        # get the token-level attributions by summing the attributions across the hidden dimension
        token_level_attributions = attributions.sum(dim=-1)  # shape: [batch_size, seq_len]
        # TODO Consider alternatives to sum. Or even summing meaning/position/context dimensions only!
        # TODO Not easy though, requires interpreting the token embeddings
        # Now take ths 0 in batch size (input sample) and print the attributions
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        for token_str, score in zip(tokens, token_level_attributions[0].tolist()):
            print(f"{token_str} => {score:.4f}")
            
    def compare_inputs(self, input_text, baseline_text, n_steps = 50, integral_approximation_method='gausslegendre'):
        # Get input ids and attention mask for that input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Get baseline input ids
        baseline = self.tokenizer(baseline_text, return_tensors="pt")
        baseline_ids = baseline["input_ids"].to(self.model.device)
        
        # Get input embeddings from ids and baseline for inputs trajectory
        with torch.no_grad():
            token_embed_layer = self.model.get_input_embeddings()
            input_embeds = token_embed_layer(input_ids)
            baseline_embeds = token_embed_layer(baseline_ids)
        
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=n_steps,
            method=integral_approximation_method,
            additional_forward_args=(attention_mask, self.model, self.tokenizer, ),    # Order is impotant here
        )
        
        # get the token-level attributions by summing the attributions across the hidden dimension
        token_level_attributions = attributions.sum(dim=-1)  # shape: [batch_size, seq_len]
        # TODO Consider alternatives to sum. Or even summing meaning/position/context dimensions only!
        # TODO Not easy though, requires interpreting the token embeddings
        # Now take ths 0 in batch size (input sample) and print the attributions
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        for token_str, score in zip(tokens, token_level_attributions[0].tolist()):
            print(f"{token_str} => {score:.4f}")