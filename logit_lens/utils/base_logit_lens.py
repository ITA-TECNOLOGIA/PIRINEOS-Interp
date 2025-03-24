from huggingface_hub import login
import matplotlib.pyplot as plt
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Tuple, Dict
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class BaseLogitLens:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model(**kwargs)
        self.model = self.wrap_model(self.model)
        self.reset_layers()

    def load_model(self, **kwargs) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """
        Loads the model and tokenizer. Must be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wraps each layer with a BlockOutputWrapper (or similar).
        Must be implemented by the subclass based on the model's structure.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def reset_layers(self):
        """
        Resets the intermediate activation wrappers.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def iterative_generation_with_layers(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"]
    ) -> Tuple[List[Dict[int, Dict[str, Tuple[str, int]]]], str]:
        """
        Generates tokens step-by-step, capturing intermediate logits from each layer
        for the requested components.
        """
        # Convert prompt to input_ids
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        generated_ids = input_ids.clone()
        all_time_steps_data = []  # To store data at each generation step

        for step in range(max_new_tokens):
            # Reset layer wrappers to capture fresh activations
            self.reset_layers()

            # Forward pass
            outputs = self.model(generated_ids)

            # Collect intermediate data from each layer
            step_data = {}
            # This assumes the model has a list or similar of layers; 
            # your subclasses should ensure this attribute is available.
            for layer_idx, layer in enumerate(self.get_layers()):
                layer_components_data = {}
                for comp in components:
                    # Retrieve the appropriate unembedded logits tensor.
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

                    # Use the logits of the last token
                    dist_last_token = dist[0, -1, :]  # shape [vocab_size]
                    probs = torch.softmax(dist_last_token, dim=-1)
                    top_prob, top_idx = probs.topk(1)
                    top_token_str = self.tokenizer.decode(top_idx)
                    top_prob_percent = int(top_prob.item() * 100)
                    layer_components_data[comp] = (top_token_str, top_prob_percent)
                step_data[layer_idx] = layer_components_data

            all_time_steps_data.append(step_data)

            # Use the final layer’s block_output distribution to choose the next token (greedy)
            final_layer = self.get_layers()[-1]
            final_dist = final_layer.block_output_unembedded[0, -1, :]
            final_probs = torch.softmax(final_dist, dim=-1)
            next_token_id = torch.argmax(final_probs).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return all_time_steps_data, final_text

    def plot_iterative_layer_logits(
        self,
        all_time_steps_data: List[Dict[int, Dict[str, Tuple[str, int]]]],
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"]
    ):
        """
        Creates subplots for each requested component with a heatmap showing
        (num_layers x num_steps). Each cell is colored by the top token's probability
        and annotated with the top token string.
        """
        num_steps = len(all_time_steps_data)
        if num_steps == 0:
            print("No data to plot.")
            return
        num_layers = len(all_time_steps_data[0])
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(components),
            figsize=(6 * len(components), 5),
            squeeze=False
        )
        axes = axes[0]

        for comp_idx, comp in enumerate(components):
            heatmap_data = np.zeros((num_layers, num_steps))
            token_labels = [["" for _ in range(num_steps)] for _ in range(num_layers)]
            for step_idx, step_data in enumerate(all_time_steps_data):
                for layer_idx in step_data.keys():
                    token_str, prob_percent = step_data[layer_idx][comp]
                    heatmap_data[layer_idx, step_idx] = prob_percent
                    token_labels[layer_idx][step_idx] = token_str

            ax = axes[comp_idx]
            im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
            ax.set_xticks(np.arange(num_steps))
            ax.set_yticks(np.arange(num_layers))
            ax.set_xticklabels([f"Step {s}" for s in range(num_steps)])
            ax.set_yticklabels([f"Layer {l}" for l in range(num_layers)])
            ax.set_title(f"Component: {comp}", fontsize=14)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            for i in range(num_layers):
                for j in range(num_steps):
                    ax.text(j, i, token_labels[i][j],
                            ha="center", va="center", color="white", fontsize=7)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability (%)")
        fig.suptitle("Iterative Generation: Top Token Probability per Layer & Step", fontsize=16)
        plt.tight_layout()
        plt.savefig('logit_lens.png')

    def get_layers(self) -> List:
        """
        A helper method to return the list of wrapped layers.
        Must be implemented by the subclass, since layer organization may differ.
        """
        raise NotImplementedError("Subclasses should implement this method.")