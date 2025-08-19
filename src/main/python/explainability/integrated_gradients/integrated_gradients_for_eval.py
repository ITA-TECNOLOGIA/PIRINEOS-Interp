import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from explainability.integrated_gradients import IGConfig


class IGEval:
    def __init__(self, model, tokenizer, forward_func: str, multiply_by_inputs=True):
        """
        Initialize the Integrated Gradients for sequence generation.
        
        Args:
            model: The model to be used for generating sequences.
            tokenizer: The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._load_ig(forward_func, multiply_by_inputs)  # Default to True, can be changed later
            
    def _load_ig(self, forward_func, multiply_by_inputs: bool):
        """
        Load the Integrated Gradients with the provided forward function.
        Can be used to reuse the model for different forward functions.
        Forward function must be defined within the class itself. 
        
        Args:
            forward_func: The forward function to be used for attribution.
        """
        if forward_func == "target_logprob_forward_func":
            selected_forward_func = self.target_logprob_forward_func
        elif forward_func == "seq_logprob_forward_func":
            selected_forward_func = self.seq_logprob_forward_func
        elif forward_func == "target_entropy_forward_func":
            selected_forward_func = self.target_entropy_forward_func
        elif forward_func == "seq_entropy_forward_func":
            selected_forward_func = self.seq_entropy_forward_func
        else:
            raise ValueError(f"Unknown forward function: {forward_func}. "
                             "Available functions: target_logprob_forward_func, "
                             "seq_logprob_forward_func, target_entropy_forward_func, "
                             "seq_entropy_forward_func.")
        self.ig = IntegratedGradients(selected_forward_func, multiply_by_inputs = multiply_by_inputs)
        
    def target_token_attributions(self, input_embeds,
                                    attention_mask,
                                    target_pos,
                                    predicted_id, 
                                    ig_config: IGConfig):
        baseline_embeds = torch.zeros_like(input_embeds)  # Baseline is zero embeddings
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_embeds,    # Use the whole sequence as input
            baselines=baseline_embeds,  # Use zero embeddings as baseline, but for the whole sequence
            n_steps=ig_config.n_steps,
            internal_batch_size = ig_config.internal_batch_size,
            method  = ig_config.integration_method,
            additional_forward_args=(attention_mask, 
                                     target_pos,
                                     predicted_id,),    # Arguments that go towards the forward function
            return_convergence_delta=ig_config.return_convergence_delta,
            target=ig_config.target,
        )
        # get the token-level attributions by summing the attributions across the hidden dimension
        token_attributions = attributions.sum(dim=-1).squeeze(0)  # shape: [seq_len]
        # 2. Compute normalized attributions (unit‐norm) and store them
        normed_tensor = token_attributions / torch.norm(token_attributions)
        return normed_tensor
    
    def seq_token_attributions(self, input_embeds,
                                attention_mask,
                                target_positions,
                                input_ids,
                                ig_config: IGConfig):
        baseline_embeds = torch.zeros_like(input_embeds)
        # Fix if there is just one target position
        if isinstance(target_positions, int):
            target_positions = [target_positions]
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_embeds,    # Use the whole sequence as input
            baselines=baseline_embeds,  # Use zero embeddings as baseline, but for the whole sequence
            n_steps=ig_config.n_steps,
            internal_batch_size = ig_config.internal_batch_size,
            method  = ig_config.integration_method,
            additional_forward_args=(attention_mask,
                                     target_positions,
                                     input_ids,),    # Arguments that go towards the forward function
            return_convergence_delta=ig_config.return_convergence_delta,
            target=ig_config.target,
        )
        # get the token-level attributions by summing the attributions across the hidden dimension
        token_attributions = attributions.sum(dim=-1).squeeze(0)  # shape: [seq_len]
        # 2. Compute normalized attributions (unit‐norm) and store them
        normed_tensor = token_attributions / torch.norm(token_attributions)
        return normed_tensor
            
    def target_logprob_forward_func(self,
                                     full_embeds,
                                     attention_mask,
                                     target_pos,
                                     predicted_id,
                                     ):
        # embeds: [1, seq_len, embed_dim]
        # Run the model with the least memory footprint possible
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask,
                                 use_case = False,
                                 output_hidden_states = False,
                                 output_attentions = False,
                                 )
        # Get logprobs for the right spot
        logits = outputs.logits[:, target_pos-1, :]         # [1, vocab_size] Use target_pos-1 as outputs are shifted
        log_probs = F.log_softmax(logits, dim=-1)         # [1, vocab_size]
        # pick out the log-prob of the predicted token
        return log_probs[:, predicted_id]                 # [1]
    
    def seq_logprob_forward_func(self,
                                full_embeds,
                                attention_mask,
                                target_positions,
                                input_ids,
                                ):
       # Run the model with the least memory footprint possible
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask,
                                 use_case = False,
                                 output_hidden_states = False,
                                 output_attentions = False,
                                 )
        
        logits  = outputs.logits                      # [1, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)     # same shape

        # gather the log-prob for each target token
        # shape: [1, len(target_positions)]
        true_logps = torch.stack([
            log_probs[:, pos-1, input_ids[pos]] # use pos-1 as outputs are shifted
            for pos in target_positions
        ], dim=1)

        # sum across the span → [1]
        return true_logps.sum(dim=1)
    
    def target_entropy_forward_func(self,
                                full_embeds,
                                attention_mask,
                                target_pos,
                                predicted_id,
                                ):
        # Run the model with the least memory footprint possible
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask,
                                 use_case = False,
                                 output_hidden_states = False,
                                 output_attentions = False,
                                 )
        
        logits   = outputs.logits                    # [1, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)    # [1, seq_len, vocab_size]
        probs     = log_probs.exp()                  # same shape

        # entropy at each position: H = -∑ p log p
        entropy = -(probs * log_probs).sum(dim=-1)   # [1, seq_len]

        # pick out only our target position → [1, 1]
        target_entropy = entropy[:, target_pos-1]

        return target_entropy    # [1]
    
    def seq_entropy_forward_func(self,
                                full_embeds,
                                attention_mask,
                                target_positions,
                                input_ids,
                                ):
        # Run the model with the least memory footprint possible
        with torch.set_grad_enabled(True):
            outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask,
                                 use_case = False,
                                 output_hidden_states = False,
                                 output_attentions = False,
                                 )
        
        logits   = outputs.logits                    # [1, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)    # [1, seq_len, vocab_size]
        probs     = log_probs.exp()                  # same shape

        # entropy at each position: H = -∑ p log p
        entropy = -(probs * log_probs).sum(dim=-1)   # [1, seq_len]

        # pick out only our target positions → [1, |T|]
        span_entropies = entropy[:, [pos - 1 for pos in target_positions]]

        # sum them for IG → [1]
        return span_entropies.sum(dim=1)