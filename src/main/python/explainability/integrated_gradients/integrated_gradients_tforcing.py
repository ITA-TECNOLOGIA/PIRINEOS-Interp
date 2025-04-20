import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from explainability.integrated_gradients import IGConfig


class IGTForcing:
    def __init__(self, model, tokenizer, forward_func = None):
        """
        Initialize the Integrated Gradients for sequence generation.
        
        Args:
            model: The model to be used for generating sequences.
            tokenizer: The tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
            
    def load_ig(self, forward_func, multiply_by_inputs: bool):
        """
        Load the Integrated Gradients with the provided forward function.
        Can be used to reuse the model for different forward functions.
        Forward function must be defined within the class itself. 
        # TODO: Change this to forward function being a wrapper class or something better.
        
        Args:
            forward_func: The forward function to be used for attribution.
        """
        self.ig = IntegratedGradients(forward_func, multiply_by_inputs = multiply_by_inputs)
        
    def compute_attributions(self, 
                             input_text, 
                             ig_config: IGConfig,):
        # Get input ids and attention mask for that input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Generate tokens using the model -> Teacher generation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample = False,
                max_new_tokens=ig_config.max_new_tokens,  # Si no, puede ser max_length
            )
        
        # Capture the additional inputs for the forward function
        prompt_length = input_ids.shape[1]
        full_length = generated_tokens.shape[1]
        forced_token_ids = generated_tokens
        
        # Get input embeddings from ids and baseline for inputs trajectory
        with torch.no_grad():
            token_embed_layer = self.model.get_input_embeddings()
            input_embeds = token_embed_layer(generated_tokens)  # Check if those are tokens
            baseline_embeds = torch.zeros_like(input_embeds)
            
        if ig_config.prediction_method == 'logprob':
            self.load_ig(self.logprob_tforcing_forward_func, ig_config.multiply_by_inputs)
        elif ig_config.prediction_method == 'entropy':
            self.load_ig(self.entropy_tforcing_forward_func, ig_config.multiply_by_inputs)
        else:
            raise ValueError(f"Unknown metric: {ig_config.prediction_method}. Supported metrics are 'logprob' and 'entropy'.")
        
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_embeds,    # Use the whole sequence as input
            baselines=baseline_embeds,  # Use zero embeddings as baseline, but for the whole sequence
            n_steps=ig_config.n_steps,
            internal_batch_size = ig_config.internal_batch_size,
            method  = ig_config.integration_method,
            additional_forward_args=(attention_mask, 
                                     prompt_length,
                                     full_length,
                                     forced_token_ids,),
            return_convergence_delta=ig_config.return_convergence_delta,
            target=ig_config.target,
        )
        
        # get the token-level attributions by summing the attributions across the hidden dimension
        input_attributions = attributions[:, :prompt_length, :]  # shape: [num_trials = 1, prompt_length, hidden_dim]
        token_level_attributions = input_attributions.sum(dim=-1)  # shape: [batch_size, full_length] #! Ojo que no solo inputs :o 
        # TODO Consider alternatives to sum. Or even summing meaning/position/context dimensions only!
        # TODO Not easy though, requires interpreting the token embeddings
        # Now take ths 0 in batch size (input sample) and print the attributions
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        print(f"Output generated: {self.tokenizer.decode(generated_tokens[0])}")
        print("Attributions:")
        for token_str, score in zip(tokens, token_level_attributions[0].tolist()):
            print(f"{token_str} => {score:.4f}")
            
    def logprob_tforcing_forward_func(self, 
        full_embeds, 
        attention_mask, 
        prompt_length,
        full_length,
        forced_token_ids,
        ):
        # Get the next token
        outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask)   
        logits = outputs.logits # shape: [batch_size, full_length, vocab_size] Genera del tamaño de toda la secuencia input+output
        
        # Get the forced token ids
        #forced_token_ids = torch.argmax(full_embeds, dim=-1)    # shape: [batch_size, full_length] Instead of batch size, it is internal batch size
        #! Me preocupa que son siempre el mismo. Esto no debería ser así, no? Debería se cada uno el que haya puesto
        #! los IG, que es la trayectoria. Más que nada porque si no, con el teacher forcing lo estoy seleccionando al
        #! final solo los tokens que son estos ids, podría verse afectado todo.
        # Al final esque como estoy con el forcing, debería ver el mismo output a lo largo de la trayectoria, no?
        # Porque seleccionar los token ids forma parte de la F también. Al menos al comparar la generación de texto
        # con los classifiers.
        
        # Get the log probs for the previous to tokens (drop the generated one)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # shape: [batch_size, full_length-1, vocab_size]
        
        gen_token_ids = forced_token_ids[:, prompt_length:full_length]    # forced_token_ids shape: [batch_size, full_length] (batch size is added by ig)
        # gen token_ids shape: [batch_size, full_length - prompt_length]

        gen_logits = log_probs[:, prompt_length-1:full_length-1, :]  # shape: [batch, num_gen_tokens, vocab_size]

        # Now pick out the log-probs of the forced tokens
        batch_size, num_gen_tokens = gen_token_ids.shape    # TODO Does this take into account the end token? A: Yes, because gen output ends at end token and we take it for all the trajectory.
        range_batch = torch.arange(batch_size).unsqueeze(-1)
        forced_token_log_probs = gen_logits[range_batch, torch.arange(num_gen_tokens), gen_token_ids]   # The first two dimensions actually could just be :
        # TODO check if this is right (?) Is it right to just select gen_token_ids or do I have to decode something?

        # Sum or average them
        avg_log_probs = forced_token_log_probs.sum(dim=-1) / num_gen_tokens # doubt this is right
        # shape: [batch_size]
        return avg_log_probs.unsqueeze(-1)
    
    def entropy_tforcing_forward_func(self, 
        full_embeds, 
        attention_mask, 
        prompt_length,
        full_length,
        forced_token_ids,
        ):
        """
        With entropy we are not interested in the forced token ids, but in the entropy of the generated tokens.
        They still are passed for forward function signature compatibility.
        """
        # Get the next token
        outputs = self.model(inputs_embeds=full_embeds, attention_mask=attention_mask)   
        logits = outputs.logits # shape: [batch_size, full_length, vocab_size] Genera del tamaño de toda la secuencia input+output
        
        # Get the entropy for the previous to tokens (drop the generated one)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # shape: [batch_size, full_length-1, vocab_size]
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # shape: [batch_size, full_length-1]
        
        gen_token_ids = forced_token_ids[:, prompt_length:full_length]    # forced_token_ids shape: [batch_size, full_length] (batch size is added by ig)
        # gen token_ids shape: [batch_size, full_length - prompt_length]

        gen_entropy = entropy[:, prompt_length-1:full_length-1]  # shape: [batch, num_gen_tokens, vocab_size]

        # Now pick out the log-probs of the forced tokens
        batch_size, num_gen_tokens = gen_token_ids.shape    # TODO Does this take into account the end token? A: Yes, because gen output ends at end token and we take it for all the trajectory.
        range_batch = torch.arange(batch_size).unsqueeze(-1)
        forced_token_entropy = gen_entropy[range_batch, torch.arange(num_gen_tokens)]   # The first two dimensions actually could just be :

        # Sum or average them
        avg_entropy = forced_token_entropy.sum(dim=-1) / num_gen_tokens # doubt this is right
        # shape: [batch_size]
        return avg_entropy.unsqueeze(-1)