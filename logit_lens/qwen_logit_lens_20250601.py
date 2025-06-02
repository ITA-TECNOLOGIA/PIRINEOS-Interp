# Ensure you’ve set HF_TOKEN in your environment before running this cell:
#   export HF_TOKEN="hf_..."

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Tuple, Dict

# ── If you want to pin to a specific GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or whichever GPU index you prefer


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn_module: torch.nn.Module):
        super().__init__()
        self.attn = attn_module
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        # 1) Run the original attention module
        output = self.attn(*args, **kwargs)
        #    output[0] is the raw attention‐mechanism output (before residual)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        # 2) Store that raw attention output for later projection
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class BlockOutputWrapper(torch.nn.Module):
    """
    Wrap one Transformer block so that after each forward pass, we stash:
      • attn_mech_output_unembedded  = lm_head(norm(attn_output_before_residual))
      • intermediate_res_unembedded   = lm_head(norm(attn_output_after_residual))
      • mlp_output_unembedded        = lm_head(norm(mlp_output))
      • block_output_unembedded      = lm_head(norm(block_output))
    """
    def __init__(self, block: torch.nn.Module, lm_head: torch.nn.Module, final_norm: torch.nn.Module):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = final_norm

        # Replace the block’s self_attn with our AttnWrapper
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        # Keep a reference to the block’s post-attention layernorm
        self.post_attention_layernorm = self.block.post_attention_layernorm

        # Placeholders for un‐embedded logits
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded  = None
        self.mlp_output_unembedded       = None
        self.block_output_unembedded     = None

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        # 1) Run the original block
        output = self.block(
            x,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        # 2) Grab the hidden_states from the block’s output
        if isinstance(output, tuple):
            hidden_states = output[0]  # shape: [batch, seq_len, d_model]
        else:
            hidden_states = output

        # 3) “Pre‐residual” attention output (un‐embedded)
        attn_output = self.block.self_attn.activations  # [batch, seq_len, d_model]
        self.attn_mech_output_unembedded = self.lm_head(self.norm(attn_output))

        # 4) Add the residual to attn_output, then project
        attn_output_with_res = attn_output + x
        self.intermediate_res_unembedded = self.lm_head(self.norm(attn_output_with_res))

        # 5) MLP on post-attn residual
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output_with_res))
        self.mlp_output_unembedded = self.lm_head(self.norm(mlp_output))

        # 6) The block’s own final output (which already includes MLP + residual)
        self.block_output_unembedded = self.lm_head(self.norm(hidden_states))

        return output

    def attn_add_tensor(self, tensor: torch.Tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded  = None
        self.mlp_output_unembedded       = None
        self.block_output_unembedded     = None

    def get_attn_activations(self):
        return self.block.self_attn.activations


class QwenLogitLens:
    """
    Load Qwen2.5-14B in FP16 (or 4-bit), wrap each block so we can extract un‐embedded logits,
    then do iterative generation, collecting per-layer logits for an “iterative logit-lens” plot.
    """
    model_name = "Qwen/Qwen2.5-14B"

    def __init__(self):
        # ── 1) HF login
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("Set HF_TOKEN in your environment before instantiating QwenLogitLens.")
        login(token=hf_token, add_to_git_credential=True)

        # ── 2) Load & wrap the model
        self.model, self.tokenizer = self.load_model(hf_token, self.model_name)
        self.model = self.wrap_model(self.model)
        self.reset_layers()

    def load_model(self, hf_token: str, model_name: str):
        # (Optional) 4-bit quantisation configuration; with 80 GB VRAM you can skip quantisation
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,               # Set to True if you want 4-bit instead of FP16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # ── Choose ONE of these two loading blocks:
        # ● Option A: full FP16 (≈28 GB VRAM)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # ● Option B: 4-bit quantised (≈6–8 GB VRAM) – uncomment if you want quantisation
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     trust_remote_code=True,
        #     quantization_config=bnb_config
        # )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer

    def wrap_model(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Replace each layer in model.model.layers with a BlockOutputWrapper so that
        after every forward pass we have access to the un-embedded logits at each subcomponent.
        """
        lm_head = model.lm_head
        final_norm = model.model.norm

        # Overwrite each block in-place
        for i, block in enumerate(model.model.layers):
            wrapped = BlockOutputWrapper(block=block, lm_head=lm_head, final_norm=final_norm)
            model.model.layers[i] = wrapped

        return model

    def reset_layers(self):
        """Clear any stored activations / cached tensors in every wrapped block."""
        for layer in self.model.model.layers:
            layer.reset()

    def iterative_generation_with_layers(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"]
    ) -> Tuple[List[Dict[int, Dict[str, Tuple[str, int]]]], str]:
        """
        Step‐by‐step generation. At each new token, run a full forward pass on prompt+so-far,
        then read out “un-embedded logits” for each requested component in each layer.
        Returns:
          • all_time_steps_data: List of length = max_new_tokens. Each item is:
                { layer_idx → { component_name → (top_token_str, prob_percent) } }
          • final_text: The complete generated text after max_new_tokens.
        Side‐effect: stores `self.all_generated_ids` (shape [1, prompt_len+max_new_tokens]) 
                     so that the plotting routine can label *every* column with its actual token.
        """
        # 1) Tokenize and move prompt to GPU
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        # Keep a copy of the original prompt length for plotting
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids.clone()
        all_time_steps_data: List[Dict[int, Dict[str, Tuple[str, int]]]] = []

        for step in range(max_new_tokens):
            # a) Reset each wrapper’s stored activations
            for layer in self.model.model.layers:
                layer.reset()

            # b) Forward pass on the entire sequence so far
            outputs = self.model(generated_ids)
            # outputs.logits is [1, seq_len, vocab_size], but we ignore it here

            # c) Collect top‐token + prob% for each requested component in each layer
            step_data: Dict[int, Dict[str, Tuple[str, int]]] = {}
            for layer_idx, layer in enumerate(self.model.model.layers):
                comp_dict: Dict[str, Tuple[str, int]] = {}
                for comp in components:
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

                    # dist has shape [1, seq_len, vocab_size]
                    dist_last = dist[0, -1, :]              # [vocab_size]
                    probs = torch.softmax(dist_last, dim=-1) # [vocab_size]
                    top_prob, top_idx = probs.topk(1)       # (value, index)

                    token_str = self.tokenizer.decode(top_idx)
                    prob_pct = int(top_prob.item() * 100)
                    comp_dict[comp] = (token_str, prob_pct)

                step_data[layer_idx] = comp_dict

            all_time_steps_data.append(step_data)

            # d) Greedy next‐token from final layer’s block_output_unembedded
            final_layer = self.model.model.layers[-1]
            final_dist = final_layer.block_output_unembedded[0, -1, :]  # [vocab_size]
            final_probs = torch.softmax(final_dist, dim=-1)
            next_token_id = torch.argmax(final_probs).unsqueeze(0).unsqueeze(0)  # [1, 1]

            # e) Append to running sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Store the final sequence of token‐IDs in the object for plotting
        self.all_generated_ids = generated_ids.detach().cpu()  # shape [1, prompt_len + max_new_tokens]

        # Decode entire sequence to text
        final_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return all_time_steps_data, final_text

    def plot_iterative_layer_logits(
        self,
        all_time_steps_data: List[Dict[int, Dict[str, Tuple[str, int]]]],
        components: List[str] = ["attn_mech_output", "intermediate_res", "mlp_output", "block_output"],
        layer_step: int = 3,
        include_input: bool = True
    ):
        """
        Draws a heatmap (one subplot per component).  We only plot every `layer_step`‐th layer
        along the y‐axis (i.e. 0, 3, 6, …).  On the x‐axis we show *all* tokens
        (prompt + generated) if include_input=True.

        Now with dynamic text color so white‐on‐white is avoided.
        """
        num_gen_steps = len(all_time_steps_data)
        if num_gen_steps == 0:
            print("No generation steps to plot.")
            return

        # Total number of hidden layers in the model
        full_num_layers = len(all_time_steps_data[0])

        # Pick which layers to actually display (0, layer_step, 2*layer_step, …)
        chosen_layer_indices = list(range(0, full_num_layers, layer_step))
        num_chosen_layers = len(chosen_layer_indices)

        # Reconstruct the full token‐ID sequence (prompt + generated)
        all_ids = self.all_generated_ids[0].tolist()  # shape: [prompt_len + num_gen_steps]
        all_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)

        prompt_len = len(all_ids) - num_gen_steps
        total_columns = len(all_ids)

        # Build one heatmap array per component: shape = (full_num_layers × total_columns)
        heatmap_per_comp: Dict[str, np.ndarray] = {}
        token_labels_full = [["" for _ in range(total_columns)] for _ in range(full_num_layers)]
        for comp in components:
            heatmap_per_comp[comp] = np.zeros((full_num_layers, total_columns), dtype=float)

        # Fill in only the “generated” columns (columns ≥ prompt_len)
        for step_idx in range(num_gen_steps):
            col_index = prompt_len + step_idx
            for layer_idx in range(full_num_layers):
                layer_data = all_time_steps_data[step_idx][layer_idx]
                for comp in components:
                    token_str, prob_pct = layer_data[comp]
                    heatmap_per_comp[comp][layer_idx, col_index] = prob_pct
                    token_labels_full[layer_idx][col_index] = token_str

        # Create one subplot per component
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(components),
            figsize=(6 * len(components), 5),
            squeeze=False
        )
        axes = axes[0]

        for comp_idx, comp in enumerate(components):
            full_map = heatmap_per_comp[comp]
            reduced_map = full_map[chosen_layer_indices, :]

            ax = axes[comp_idx]

            # Use "Blues_r" so that 0→dark navy, 100→almost white
            im = ax.imshow(reduced_map, cmap="Blues_r", aspect="auto", vmin=0, vmax=100)

            # X‐axis: label every column with the actual token (prompt + generated)
            ax.set_xticks(np.arange(total_columns))
            ax.set_xticklabels(all_tokens, rotation=45, ha="right", fontsize=8)

            # Y‐axis: only show labels for chosen layers
            y_labels = [f"Layer {i}" for i in chosen_layer_indices]
            ax.set_yticks(np.arange(num_chosen_layers))
            ax.set_yticklabels(y_labels, fontsize=8)

            ax.set_title(f"LogitLens Heatmap on Qwen2.5-14B, for 'The capital of the US'", fontsize=10)

            # Annotate each cell with the token predicted at that layer/step,
            # but choose text color based on how light/dark the cell is.
            for i, layer_i in enumerate(chosen_layer_indices):
                for j in range(total_columns):
                    if j >= prompt_len:
                        txt = token_labels_full[layer_i][j]
                        val = reduced_map[i, j]  # this is the probability (%) in [0..100]
                        text_color = "black" if val > 50 else "white"
                        ax.text(
                            j, i,
                            txt,
                            ha="center", va="center",
                            color=text_color,
                            fontsize=6
                        )

            # Colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability (%)")

        #fig.suptitle("Iterative Generation: Top Token Prob per Layer & Step", fontsize=16)
        plt.tight_layout()
        plt.savefig("logit_lens_every3_blues_dynamic_text.png")
        plt.show()



if __name__ == "__main__":
    # ── Instantiate helper
    qwen_lens = QwenLogitLens()

    input_prompt = "The capital of the US"
    # We’ll only track “intermediate_res” here, but you can pass multiple components in a list
    components_to_capture = ["intermediate_res"]

    all_time_steps_data, final_output = qwen_lens.iterative_generation_with_layers(
        prompt=input_prompt,
        max_new_tokens=3,
        components=components_to_capture
    )

    # Now plot, subsampling every 3 layers and including all tokens on the x‐axis
    qwen_lens.plot_iterative_layer_logits(
        all_time_steps_data,
        components=components_to_capture,
        layer_step=3,
        include_input=True
    )

    print("Final generated text:\n", final_output)
