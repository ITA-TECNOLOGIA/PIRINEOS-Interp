from unsloth import FastLanguageModel

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import matplotlib.pyplot as plt

import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens.plotting import PredictionTrajectory
import plotly.io as pio
from plotly.subplots import make_subplots

class QwenTunedLens:
    
    access_token_write = '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'
    model_name = "trained_models/checkpoint-100"
    trained_lens = "tuned-lens-ITA/my_lenses/Qwen2/Qwen25-7B-Instruct"
    
    def __init__(self, device_map = "auto"):
        self.model, self.tokenizer = self.load_model(self.access_token_write, self.model_name, device_map)
        
        # Initialize lenses
        self.tuned_lens = TunedLens.from_model_and_pretrained(self.model, self.trained_lens)
        self.tuned_lens = self.tuned_lens.to(device_map)
        self.logit_lens = LogitLens.from_model(self.model)
    
    def load_model(self, access_token_write, model_name, device_map):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map={'': 0}, torch_dtype=torch.float16) 
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
    def explain(
        self, 
        prompt, 
        statistic="entropy", 
        lens_type="tuned", 
        layer_stride=2, 
        token_range=None
    ):
        """
        Generate an explainability plot for a given prompt using a specified lens and statistic.
        
        Parameters:
            prompt (str): The input text prompt.
            statistic (str): Which statistic to compute. Options include 'entropy', 'cross_entropy', 'forward_kl'.
            lens_type (str): Which lens to use ('tuned' for TunedLens, 'logit' for LogitLens).
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
        
        # Select the appropriate lens
        if lens_type.lower() == "tuned":
            selected_lens = self.tuned_lens
        elif lens_type.lower() == "logit":
            selected_lens = self.logit_lens
        else:
            raise ValueError("Invalid lens_type. Choose 'tuned' or 'logit'.")
        
        # Build the prediction trajectory and slice the sequence
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=selected_lens,
            model=self.model,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            targets=targets,
        ).slice_sequence(slice(*token_range))
        
        # Compute the chosen statistic, apply layer stride, and create the figure
        fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
            title=f"{selected_lens.__class__.__name__} ({self.model.name_or_path}) {statistic}"
        )
        return fig
    
    def explain_output(
        self,
        prompt,
        statistic="entropy",
        lens_type="tuned",
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
            lens_type (str): Which lens to use ('tuned' for TunedLens or 'logit' for LogitLens).
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
        
        # Select the appropriate lens.
        if lens_type.lower() == "tuned":
            selected_lens = self.tuned_lens
        elif lens_type.lower() == "logit":
            selected_lens = self.logit_lens
        else:
            raise ValueError("Invalid lens_type. Choose 'tuned' or 'logit'.")
        
        # Build the prediction trajectory using the generated output.
        pred_traj = PredictionTrajectory.from_lens_and_model(
            lens=selected_lens,
            model=self.model,
            input_ids=generated_ids,
            tokenizer=self.tokenizer,
            targets=targets,
        ).slice_sequence(slice(*token_range))
        
        # Compute the chosen statistic, apply layer stride, and create the figure.
        fig = getattr(pred_traj, statistic)().stride(layer_stride).figure(
            title=f"{selected_lens.__class__.__name__} ({self.model.name_or_path}) {statistic} for Generated Output"
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
    qwen_lens = QwenTunedLens(device_map="cuda:0")
    
    # Generate an explainability plot for a given INPUT
    #fig = qwen_lens.explain(
    #    prompt="it was the best of times, it was the worst of times",
    #    statistic="entropy",
    #    lens_type="logit",
    #    layer_stride=2,
    #    token_range=(0, 20)
    #)
    
    # Problema 1 -> Answer: 5
    prompt_1 = "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"
    
    # Generate an explainability plot for the GENERATED OUTPUT
    fig = qwen_lens.explain_output(
        prompt=prompt_1,
        statistic="entropy",
        lens_type="tuned",
        layer_stride=2,
        token_range=(0, 30),
        generation_kwargs={"max_new_tokens": 100, "do_sample": True, "temperature": 0.8}
    )
    
    fig.write_html("explainability_plot_problema1_razonador.html")