from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import matplotlib.pyplot as plt
import torch
import plotly.io as pio
from plotly.subplots import make_subplots

from tuned_lens.nn.lenses import TunedLens as ARTunedLens
from tuned_lens.plotting import PredictionTrajectory

import os
import psutil
import time
import numpy as np

# Call this function wherever you want to log CPU and RAM usage.
def log_resource_usage(label=""):
    process = psutil.Process(os.getpid())
    # CPU percent since last call; use a small interval to get an updated snapshot
    cpu_pct = process.cpu_percent(interval=0.1)
    # Resident Set Size (RSS) in bytes → convert to GB
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"[RESOURCE] {label} | CPU%: {cpu_pct:.2f}% | RAM: {mem_gb:.2f} GB")

class TunedLens:
    
    def __init__(self, model, tokenizer, lens):
        """
        Initialize the TunedLens class with a model, tokenizer, and lens.
        Model and tokenizer can come from DefaultHFModelLoader.
        Lens can be a location for the lens or one of the pretrained ones from tuned lens.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize lenses
        self.tuned_lens = ARTunedLens.from_model_and_pretrained(self.model, lens)
        self.tuned_lens = self.tuned_lens.to(self.model.device)
    
    def explain(
        self, 
        prompt, 
        statistic="entropy", 
        layer_stride=2, 
        token_range=None
    ):
        """
        Generate an explainability plot for a given prompt using a specified lens and statistic.
        
        Parameters:
            prompt (str): The input text prompt.
            statistic (str): Which statistic to compute. Options include 'entropy', 'cross_entropy', 'forward_kl'.
            layer_stride (int): The stride to use when plotting the layers.
            token_range (tuple or list): A two-element tuple/list specifying the start and end token indices.
                                         If None, the full token sequence is used.
                                         
        Returns:
            fig: A Plotly figure object representing the explainability plot.
        """
        # Tokenize input and prepare targets
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            raise ValueError("Please provide some input text.")
        targets = input_ids[1:] + [self.tokenizer.eos_token_id]
        
        # Determine token range
        if token_range is None:
            token_range = (0, len(input_ids))
        elif token_range[0] == token_range[1]:
            raise ValueError("Please provide a valid token range (start and end must differ).")
        
        # Build the prediction trajectory and slice the sequence
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=self.tuned_lens,
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            targets=targets,
        ).slice_sequence(slice(*token_range))
        
        # Compute the chosen statistic, apply layer stride, and create the figure
        fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
            title=f"{self.tuned_lens.__class__.__name__} ({self.model.name_or_path}) {statistic}"
        )
        return fig
    
    def stats_explain(
        self, 
        input_ids = None, # Pass it the token ids directly
        prompt = None, 
        statistic="entropy", 
        layer_stride=2, 
        token_range=None
    ):
        """
        Instead of generating the explainability plot, it returns the resulting statistics.
        
        Parameters:
            prompt (str): The input text prompt.
            statistic (str): Which statistic to compute. Options include 'entropy', 'cross_entropy', 'forward_kl'.
            layer_stride (int): The stride to use when plotting the layers.
            token_range (tuple or list): A two-element tuple/list specifying the start and end token indices.
                                         If None, the full token sequence is used.
                                        
        """
        # Tokenize input and prepare targets
        if input_ids is None:
            input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0 and prompt is None:
            raise ValueError("Please provide some input text.")
        targets = input_ids[1:] + [self.tokenizer.eos_token_id]
        
        # Determine token range
        if token_range is None:
            token_range = (0, len(input_ids))
        elif token_range[0] == token_range[1]:
            raise ValueError("Please provide a valid token range (start and end must differ).")
        
        #log_resource_usage(f"before predictiontrajectory")
        
        # Build the prediction trajectory and slice the sequence
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=self.tuned_lens,
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            targets=targets,
        ).slice_sequence(slice(*token_range))
        
        #log_resource_usage(f"after predictiontrajectory")
        
        return pred_traj
        
        # Compute the chosen statistic, apply layer stride, and create the figure
        fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
            title=f"{self.tuned_lens.__class__.__name__} ({self.model.name_or_path}) {statistic}"
        )
        return fig
    
    def stats_explain_minimal(
    self,
    input_ids: list[int],
    token_range: tuple[int, int],
    layer_stride: int = 1,
    ):
        """
        Compute only:
        • log_probs[layer_i][token_pos][:]   (as a NumPy array) for each kept layer
        • entropy[layer_i][token_pos]         (NumPy array)
        • forward_kl[layer_i][token_pos]      (NumPy array)
        **Without** ever materializing the full (layers x seq_len x vocab) tensor in CPU/RAM.

        Args:
            tuned_lens:       your TunedLens object (with lens.forward)
            model:            the HF model (already .eval() and on correct device)
            tokenizer:        the HF tokenizer
            input_ids:        full sequence of token IDs (prompt + generation)
            token_range:      (start_idx, end_idx) within input_ids for which we want stats
            layer_stride:     only keep every `layer_stride`‐th layer (e.g. 2, 4)

        Returns:
            {
                "log_probs": np.ndarray shape (num_kept_layers, slice_len, vocab_size),
                "entropy":   np.ndarray shape (num_kept_layers, slice_len),
                "forward_kl": np.ndarray shape (num_kept_layers, slice_len)
            }
        """

        # 1) Run the model on the full input_ids, get hidden_states + final logits
        model_device = next(self.model.parameters()).device
        input_th = torch.tensor(input_ids, device=model_device).unsqueeze(0)  # (1, seq_len)

        # We need both hidden_states and logits
        # Make sure your model returns hidden_states:
        model_out = self.model(input_th, output_hidden_states=True)
        hidden_states = model_out.hidden_states  # tuple of length (num_layers+1); each is (1, seq_len, d_model)
        logits_full = model_out.logits.squeeze(0).detach()  # (seq_len, vocab_size)

        # 2) Prepare slices
        start_tok, end_tok = token_range
        slice_len = end_tok - start_tok
        vocab_size = logits_full.shape[-1]

        # Convert the model's logits over that slice to log‐probs, for forward_KL
        model_lps_slice = (
            logits_full[start_tok:end_tok, :]
            .log_softmax(dim=-1)   # still a torch.Tensor shape (slice_len, vocab_size)
            .detach()
            .cpu()
            .numpy()
        )
        
        probs_model_slice = np.exp(model_lps_slice)  # NumPy array (slice_len, vocab)
        ent_model = -(probs_model_slice * model_lps_slice).sum(axis=-1)  # (slice_len,)
        # We can delete `logits_full` on GPU now if it's still occupying memory
        del logits_full

        # 3) Loop over every kept layer, one at a time
        all_layer_log_probs = []
        all_layer_entropy   = []
        all_layer_forwardkl = []

        num_layers_total = len(hidden_states) - 1  # last hidden_states[-1] is post‐embedding, not needed for lens

        for layer_idx in range(0, num_layers_total, layer_stride):
            # Log memory before we process this layer
            #log_resource_usage(f"before processing layer {layer_idx}")

            # 3.1) Extract that layer's hidden state, slice to token_range
            # hidden_states[layer_idx]: shape (1, seq_len, d_model)
            h_layer: torch.Tensor = hidden_states[layer_idx].squeeze(0)  # (seq_len, d_model)
            h_slice = h_layer[start_tok:end_tok, :].unsqueeze(0)        # (1, slice_len, d_model)

            # 3.2) Run tuned_lens.forward() on this slice to get logits over vocab
            #        Output shape: (1, slice_len, vocab_size)
            proj_logits = self.tuned_lens.forward(h_slice, layer_idx)  # still on model_device

            # 3.3) Compute log_probs (slice_len × vocab), entropy, and forward KL
            # Move to CPU/double‐check dtype
            proj_lps = proj_logits.log_softmax(dim=-1).squeeze(0)  # (slice_len, vocab_size) on GPU
            proj_probs = proj_lps.exp()                            # (slice_len, vocab_size)

            # Entropy per token: -∑ p * log p
            #   shape → (slice_len,)
            ent = -(proj_probs * proj_lps).sum(dim=-1)  # still on GPU

            # Forward KL: ∑ p_model * (log p_model − log p_lens)
            #   p_model = exp(model_lps_slice) but we already have model_lps_slice on CPU
            #   proj_lps on GPU, so bring proj_lps to CPU
            proj_lps_cpu = proj_lps.detach().cpu()        # (slice_len, vocab_size)
            # model_lps_slice: (slice_len, vocab_size) NumPy
            #   → convert to torch Tensor on CPU so we can do operations easily
            model_lps_cpu = torch.from_numpy(model_lps_slice)  # (slice_len, vocab_size)
            kl = ( model_lps_cpu.exp() * (model_lps_cpu - proj_lps_cpu) ).sum(dim=-1)  # (slice_len,)

            # 3.4) Convert each to NumPy, append to our lists
            all_layer_log_probs.append(proj_lps_cpu.numpy())  # each is (slice_len, vocab_size)
            all_layer_entropy.append(ent.detach().cpu().numpy())    # each is (slice_len,)
            all_layer_forwardkl.append(kl.detach().cpu().numpy())   # each is (slice_len,)

            # 3.5) Clean up intermediate tensors for this layer
            del h_slice, proj_logits, proj_lps, proj_probs, proj_lps_cpu, ent, model_lps_cpu, kl
            torch.cuda.empty_cache()
            #log_resource_usage(f"after finishing layer {layer_idx}")
            
        # 4) NOW append the final “model” layer as one more row:
        all_layer_log_probs.append(model_lps_slice)     # shape (slice_len, vocab)
        all_layer_entropy.append(ent_model)             # shape (slice_len,)
        # Forward KL of model vs. itself = 0 for every token
        all_layer_forwardkl.append(np.zeros(slice_len, dtype=np.float32))

        # 4) Stack into big NumPy arrays
        #    log_probs:   shape (num_kept_layers, slice_len, vocab_size)
        #    entropy:     shape (num_kept_layers, slice_len)
        #    forward_kl:  shape (num_kept_layers, slice_len)
        log_probs_arr   = np.stack(all_layer_log_probs, axis=0)
        entropy_arr     = np.stack(all_layer_entropy,   axis=0)
        forwardkl_arr   = np.stack(all_layer_forwardkl, axis=0)

        # 5) Clean up the hidden_states tuple (so GPU/CPU RAM frees)
        del hidden_states
        torch.cuda.empty_cache()

        return {
            "log_probs":   log_probs_arr,
            "entropy":     entropy_arr,
            "forward_kl":  forwardkl_arr,
        }
    
    def explain_output(
        self,
        prompt,
        statistic="entropy",
        layer_stride=2,
        token_range=None,
        generation_kwargs=None
    ):
        """
        Generate an explainability plot for the generated output.
        
        Instead of using the prompt tokens as target (shifted), this method:
          1. Generates an output from the model using the prompt.
          2. Uses the generated output tokens as the sequence to analyze.
          3. Computes the specified statistic over the prediction trajectory.
        
        Parameters:
            prompt (str): The input text prompt.
            statistic (str): Which statistic to compute (e.g. 'entropy', 'cross_entropy', 'forward_kl').
            layer_stride (int): Layer stride for the prediction trajectory.
            token_range (tuple or list): A two-element tuple/list specifying the start and end indices 
                                         for slicing the generated sequence.
                                         If None, the full generated sequence is used.
            generation_kwargs (dict): Optional dictionary of generation parameters to pass to `model.generate`.
                                      For example, {"max_new_tokens": 50, "do_sample": True, "temperature": 0.8}.
        
        Returns:
            fig: A Plotly figure object representing the explainability plot for the generated output.
        """
        # Set default generation parameters if none are provided.
        if generation_kwargs is None:
            generation_kwargs = {"max_new_tokens": 50}
        
        # Tokenize the prompt and move to the model's device.
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate the output from the model.
        generated = self.model.generate(input_ids, **generation_kwargs)
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(generated_text)
        
        # Convert generated output (assumed batch size 1) to a list of token ids.
        generated_ids = generated[0].tolist()

        # Determine token range: if not provided, use the full generated sequence.
        if token_range is None:
            token_range = (0, len(generated_ids))
        elif token_range[0] == token_range[1]:
            raise ValueError("Please provide a valid token range (start and end must differ).")
        
        # Create target tokens by shifting the generated token sequence.
        targets = generated_ids[1:] + [self.tokenizer.eos_token_id]
        
        # Build the prediction trajectory using the generated output.
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=self.tuned_lens,
            model=self.model,
            input_ids=generated_ids,
            tokenizer=self.tokenizer,
            targets=targets,
        ).slice_sequence(slice(*token_range))
        
        # Compute the chosen statistic, apply layer stride, and create the figure.
        fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
            title=f"{self.tuned_lens.__class__.__name__} ({self.model.name_or_path}) {statistic} for Generated Output"
        )
        return fig
    
    """def generate_trajectory(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        
        third_line = slice(14, 21)

        predictition_traj_ring = PredictionTrajectory.from_lens_and_model(
            self.tuned_lens,
            self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            targets=targets_ring,
        )#.slice_sequence(third_line) # Slice the sequence to select layers"""
        
        
        
    

if __name__ == "__main__":
    qwen_lens = TunedLens(device_map="cuda:0")
    
    # Generate an explainability plot for a given INPUT
    #fig = qwen_lens.explain(
    #    prompt="it was the best of times, it was the worst of times",
    #    statistic="entropy",
    #    layer_stride=2,
    #    token_range=(0, 20)
    #)
    
    # Problema 1 -> Answer: 5
    prompt_1 = "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
    
    # Generate an explainability plot for the GENERATED OUTPUT
    fig = qwen_lens.explain_output(
        prompt=prompt_1,
        statistic="entropy",
        layer_stride=2,
        token_range=(0, 30),
        generation_kwargs={"max_new_tokens": 100, "do_sample": True, "temperature": 0.8}
    )
    fig.write_html("explainability_plot_problema1.html")