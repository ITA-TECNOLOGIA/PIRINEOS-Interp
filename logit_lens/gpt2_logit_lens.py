from huggingface_hub import login
import matplotlib.pyplot as plt
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Tuple, Dict
import os

import numpy as np


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
    #! tengo la corrección en el gpt
    def __init__(self, block, lm_head, norm):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        
        # Wrap attention module
        # Take into account that this wraps just the attn, leaving the ln_1 as it it,
        # so we don't need to take into account the ln_1 too
        self.block.attn = AttnWrapper(self.block.attn)
        self.post_attention_layernorm = self.block.ln_2
        
        # Store intermediate activations
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        
    def forward(self, x, attention_mask=None, **kwargs):
        output = self.block(x, attention_mask=attention_mask, 
                          **kwargs)
        """
        def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
        """
        
        # Get hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Get attention output and compute intermediate activations
        attn_output = self.block.attn.activations
        self.attn_mech_output_unembedded = self.lm_head(self.norm(attn_output))
        
        # Add residual connection
        attn_residual = attn_output + x
        self.intermediate_res_unembedded = self.lm_head(self.norm(attn_residual)) # Fíjate que hace bien lo de meter el norm y la head del final
        
        # MLP output
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_residual))
        self.mlp_output_unembedded = self.lm_head(self.norm(mlp_output))
        
        # Block output
        # In the case of prenorm like gpt2, it should coincide with mlp_output
        self.block_output_unembedded = self.lm_head(self.norm(hidden_states))   

        return output

    def attn_add_tensor(self, tensor):
        self.block.attn.add_tensor = tensor

    def reset(self):
        self.block.attn.reset()
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def get_attn_activations(self):
        return self.block.attn.activations


class GPT2LogitLens:
    
    access_token_write = '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'
    model_name = "gpt2"
    
    def __init__(self):
        self.access_token_write = os.getenv("HF_TOKEN")
        self.model, self.tokenizer = self.load_model(self.access_token_write, self.model_name)
        self.model = self.wrap_model(self.model)
        self.reset_layers()
    
    def load_model(self, access_token_write, model_name):
        login(
          token=access_token_write,
          add_to_git_credential=True
        )
        # Quantization config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True, #This is required to load Qwen models.
            quantization_config=bnb_config # Apply 4-bit quantization
        )
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer
    
    def wrap_model(self, model):
        # Meto el wrapper en el modelo
        for i, layer in enumerate(model.transformer.h):
            self.model.transformer.h[i] = BlockOutputWrapper(
                layer,
                model.lm_head,
                model.transformer.ln_f
            )
        return model

    def reset_layers(self):
        for layer in self.model.transformer.h:
            layer.reset()
    
    def iterative_generation_with_layers(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"]
    ) -> Tuple[List[Dict[int, Dict[str, Tuple[str, int]]]], str]:
        """
        Generates tokens step-by-step, capturing intermediate logits from each layer
        for the requested components (e.g. 'attn_mech_output', 'mlp_output', etc.).
        
        Args:
            model: A transformers model with BlockOutputWrapper layers.
            tokenizer: The corresponding tokenizer.
            prompt (str): The initial prompt string.
            max_new_tokens (int): How many tokens to generate step-by-step.
            components (List[str]): Which activation components to record.
                Must be among:
                    ['attn_mech_output', 'intermediate_res', 'mlp_output', 'block_output'].

        Returns:
            all_time_steps_data: A list of length = max_new_tokens.
                Each element is a dict: { layer_idx -> { component_name -> (top_token, prob_percent) } }
            final_text: The final generated text after `max_new_tokens` steps.
        """
        # Convert prompt to input_ids
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        generated_ids = input_ids.clone()

        # This will store data for each generation step
        # all_time_steps_data[step] = {layer_idx -> {component -> (token, prob%)}}
        all_time_steps_data = []

        for step in range(max_new_tokens):
            # 1) Reset wrappers so we capture fresh activations
            for layer in self.model.transformer.h:
                layer.reset()

            # 2) Forward pass on the entire current input
            outputs = self.model(generated_ids) #! realmente parece que solo le mete el hidden state, no se pa que meterle tanta cosa en el forward del block

            # 3) Collect intermediate data from each layer
            step_data = {}
            for layer_idx, layer in enumerate(self.model.transformer.h):
                # Create a dict for each layer with all requested components
                layer_components_data = {}

                for comp in components:
                    # For each component, get the final distribution over vocab
                    # shape: [batch=1, seq_len, vocab_size]
                    if comp == "attn_mech_output":
                        dist = layer.attn_mech_output_unembedded
                    elif comp == "intermediate_res":
                        dist = layer.intermediate_res_unembedded
                    elif comp == "mlp_output":
                        dist = layer.mlp_output_unembedded
                    elif comp == "block_output":
                        dist = layer.block_output_unembedded
                    else:
                        raise ValueError(f"Unrecognized component: {comp}")

                    # We only care about the last token
                    dist_last_token = dist[0, -1, :]  # shape [vocab_size]
                    probs = torch.softmax(dist_last_token, dim=-1)

                    # Get top token
                    top_prob, top_idx = probs.topk(1)
                    top_token_str = self.tokenizer.decode(top_idx)
                    top_prob_percent = int(top_prob.item() * 100)

                    layer_components_data[comp] = (top_token_str, top_prob_percent)

                step_data[layer_idx] = layer_components_data

            all_time_steps_data.append(step_data)

            # 4) Decide on the next token from the *final layer’s* block_output distribution (greedy)
            final_layer = self.model.transformer.h[-1]
            final_dist = final_layer.block_output_unembedded[0, -1, :]
            final_probs = torch.softmax(final_dist, dim=-1)
            next_token_id = torch.argmax(final_probs).unsqueeze(0).unsqueeze(0)

            # 5) Append next token to the running sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Decode the final text
        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return all_time_steps_data, final_text
    
    def plot_iterative_layer_logits(
        self,
        all_time_steps_data: List[Dict[int, Dict[str, Tuple[str, int]]]],
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"]
    ):
        """
        Creates subplots for each requested component, with a heatmap showing
        (num_layers x num_steps). Each cell is colored by the top token's probability
        and annotated with the top token string.

        Args:
            all_time_steps_data: A list (length = # generation steps). Each item is:
                {layer_idx -> {component_name -> (token_str, prob%)}}
            components: Which components to visualize in separate subplots.
        """
        num_steps = len(all_time_steps_data)
        # Assume each step_data has the same set of layer keys
        # We'll infer the number of layers from the first step
        if num_steps == 0:
            print("No data to plot.")
            return
        num_layers = len(all_time_steps_data[0])

        # We'll create one subplot per component
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(components),
            figsize=(6 * len(components), 5),
            squeeze=False
        )
        axes = axes[0]  # flatten the 2D array of axes to 1D

        for comp_idx, comp in enumerate(components):
            # Prepare a matrix for the probabilities and a matrix for the tokens
            heatmap_data = np.zeros((num_layers, num_steps))
            token_labels = [["" for _ in range(num_steps)] for _ in range(num_layers)]

            # Fill in the data
            for step_idx, step_data in enumerate(all_time_steps_data):
                for layer_idx in step_data.keys():
                    token_str, prob_percent = step_data[layer_idx][comp]
                    heatmap_data[layer_idx, step_idx] = prob_percent
                    token_labels[layer_idx][step_idx] = token_str

            # Create the heatmap for this component
            ax = axes[comp_idx]
            im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")

            # Axis labels
            ax.set_xticks(np.arange(num_steps))
            ax.set_yticks(np.arange(num_layers))
            ax.set_xticklabels([f"Step {s}" for s in range(num_steps)])
            ax.set_yticklabels([f"Layer {l}" for l in range(num_layers)])
            ax.set_title(f"Component: {comp}", fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            # Annotate each cell with the top token
            for i in range(num_layers):
                for j in range(num_steps):
                    ax.text(
                        j, i,
                        token_labels[i][j],
                        ha="center", va="center",
                        color="white", fontsize=7
                    )

            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability (%)")

        fig.suptitle("Iterative Generation: Top Token Probability per Layer & Step", fontsize=16)
        plt.tight_layout()
        plt.savefig('logit_lens.png')


# Example usage:
if __name__ == "__main__":
    input_text = "The capital of the US is"
    gpt3_abstract = """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training
    on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic
    in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of
    thousands of examples. By contrast, humans can generally perform a new language task from only
    a few examples or from simple instructions – something which current NLP systems still largely
    struggle to do. Here we show that scaling up language models greatly improves task-agnostic,
    few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion
    parameters, 10x more than any previous non-sparse language model, and test its performance in
    the few-shot setting.""".replace("\n", " ")
    components_to_capture = ["attn_mech_output"]

    # For GPT-2:
    gpt2_logit_lens = GPT2LogitLens()
    all_time_steps_data, final_text = gpt2_logit_lens.iterative_generation_with_layers(
        prompt=input_text,
        max_new_tokens=5,
        components=components_to_capture
    )
    gpt2_logit_lens.plot_iterative_layer_logits(all_time_steps_data, components=components_to_capture)
    print("GPT-2 final text:", final_text)