#!/usr/bin/env python
# coding: utf-8

# # GSM8K dataset analysis

# In[1]:


from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import json, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from eval_gsm8k.utils import extract_predicted_answer

os.chdir('..')

access_token = "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"
login(token=access_token, add_to_git_credential=True)


# In[2]:


# METADATA
results_paths = {
    "Qwen2.5_BASE": 'eval_results/GSM8K_Qwen__Qwen2.5-14B_2025-06-08_09-10-05',
    "Qwen2.5_GRPO": 'eval_results/GSM8K_outputs__Qwen2_5_14B_GRPO__checkpoint-1500_2025-06-08_09-14-30',
    "Qwen2.5_distil": 'eval_results/GSM8K_deepseek-ai__DeepSeek-R1-Distill-Qwen-14B_2025-06-08_23-36-11',
}

run_paths = {
    "base": results_paths["Qwen2.5_BASE"],
    "grpo": results_paths["Qwen2.5_GRPO"],
    "distil": results_paths["Qwen2.5_distil"],
}

# load all the tokenizers
tokenizers = {
    "base": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B"),
    "grpo": AutoTokenizer.from_pretrained("outputs/Qwen2_5_14B_GRPO/checkpoint-1500"),
    "distil": AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
}


# In[3]:


# MISCELANEOUS FUNCTIONS
def recover_index(run_dir):
    """
    Recover the index from the results directory.
    """
    index_path = Path(run_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    return pd.DataFrame(index_data)

def print_accuracy(df):
    """
    Print the accuracy of the model based on the DataFrame.
    """
    total = len(df)
    correct = df['correct'].sum()
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

def get_idxs(row, tokenizer):
    """
    Get the indices of the tokens that correspond to the predicted answer in the text answer.
    """
    # Get data from row
    text_answer = row['model_answers_text'][0]
    predicted_answer = row['extracted_model_answers'][0]

    # Search for the last predicted answer in the text answer
    start_char = text_answer.rfind(predicted_answer)
    if start_char == -1:
        return []

    # Get where it ends
    end_char = start_char + len(predicted_answer)
    enc = tokenizer(text_answer, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    
    # Get all the idxs
    idxs = [
        i for i, (start, end) in enumerate(offsets)
        if not (end <= start_char or start >= end_char)
    ]
    
    return idxs


# In[9]:


# RESULTS FUNCTIONS

def extract_entropy_data(row, run_dir, tokenizer):
    """
    layerwise_entropy: [num_layers]
    overall_entropy: scalar
    """
    idxs = get_idxs(row, tokenizer)
    arr = torch.load(os.path.join(run_dir, row["entropy_file"]), weights_only=False)
    # If no final answer, return nan array
    if not idxs:
        num_layers = arr.shape[0]
        return torch.full((num_layers,), float('nan')), float('nan')
    # Now I get the entropy values for the idxs
    result_entropy = arr[:,idxs]
    # avg over tokens
    result_layerwise_entropy = result_entropy.mean(axis=1)
    # avg everything
    result_auc = result_layerwise_entropy.sum().item()
    # Get input length
    input_tokens = tokenizer.encode(row['question'], add_special_tokens=False)
    input_len = len(input_tokens)
    # Get reasoning tokens mean entropy
    reasoning_layerwise_entropy = arr[:, input_len:idxs[0]].mean(axis=1)
    reasoning_auc = reasoning_layerwise_entropy.sum().item()  # AUC is sum of the layerwise entropies
    # Now for the last layer
    last_result_entropy = arr[-1, idxs].mean().item()
    last_reasoning_entropy = arr[-1, input_len:idxs[0]].mean().item()
    return result_layerwise_entropy, result_auc, reasoning_layerwise_entropy, reasoning_auc, last_result_entropy, last_reasoning_entropy

def extract_target_avg_logp(row, run_dir):
    """
    Extract the average log probability of the target answer from the row data.
    output shape: [num_layers]
    overall_logp: scalar
    """
    arr = torch.load(os.path.join(run_dir, row["layer_avg_logp_file"]), weights_only=False)
    result_auc = arr.sum().item()  # AUC is sum of the layerwise log probabilities
    # Now the last layer
    last_result = arr[-1].item()
    return arr, result_auc, last_result

def extract_forward_kl_data(row, run_dir, tokenizer):
    """
    layerwise_forward_kl: [num_layers]
    overall_forward_kl: scalar
    """
    idxs = get_idxs(row, tokenizer)
    arr = torch.load(os.path.join(run_dir, row["forward_kl_file"]), weights_only=False)
    # If no final answer, return nan array
    if not idxs:
        num_layers = arr.shape[0]
        return torch.full((num_layers,), float('nan')), float('nan')
    # Now I get the forward KL values for the idxs
    result_forward_kl = arr[:, idxs]
    # avg over tokens
    result_layerwise_fkl = result_forward_kl.mean(axis=1)
    # avg everything
    result_auc = result_layerwise_fkl.mean()
    # Get input length
    input_tokens = tokenizer.encode(row['question'], add_special_tokens=False)
    input_len = len(input_tokens)
    # get reasoning tokens mean forward KL
    reasoning_layerwise_fkl = arr[:, input_len:idxs[0]].mean(axis=1)
    reasoning_auc = reasoning_layerwise_fkl.sum().item()
    # Now for the last layer
    last_result_fkl = arr[-1, idxs].mean().item()
    last_reasoning_fkl = arr[-1, input_len:idxs[0]].mean().item()
    return result_layerwise_fkl, result_auc, reasoning_layerwise_fkl, reasoning_auc, last_result_fkl, last_reasoning_fkl

def load_topk(row, run_dir, what="final"):
    """Return a dict with keys logp / ids / toks for 'final' or 'layer'."""
    prefix = "topk_final_" if what == "final" else "topk_layer_"
    logp   = torch.load(os.path.join(run_dir, row[f"{prefix}log_probs_file"]),
                        weights_only=False)
    ids    = torch.load(os.path.join(run_dir, row[f"{prefix}ids_file"]),
                        weights_only=False)
    toks   = torch.load(os.path.join(run_dir, row[f"{prefix}tokens_file"]),
                        weights_only=False)
    return dict(logp=logp, ids=ids, toks=toks)

def summarise_final_topk(topk_dict, answer_token_ids):
    """
    * margin:   mean(logp_rank1 - logp_rank2)          → confidence
    Es cuánto de margen hay entre el primero y el último
    """
    logp = topk_dict["logp"]           # (T, K)
    ids  = topk_dict["ids"]            # (T, K)
    T, K = logp.shape          # ← extract T (positions) and K (top-K)

    rank1 = logp[:, 0]
    rank2 = logp[:, 1]
    margin = (rank1 - rank2).mean().item()

    return dict(final_margin      = margin)


# In[10]:


# Stats functions
def descriptive_stats(df, metric):
    grp = df.groupby('run')[metric]
    stats = grp.agg(
        mean   = 'mean',
        median = 'median',
        Q1    = lambda x: x.quantile(0.25),
        Q3    = lambda x: x.quantile(0.75),
        min    = 'min',
        max    = 'max',
    )
    return stats

def cliffs_delta(x, y):
    """Compute Cliff’s delta effect size between two 1D arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    n1, n2 = len(x), len(y)
    # count how many times xi > yj and xi < yj
    greater = sum(np.sum(xi > y)   for xi in x)
    lesser  = sum(np.sum(xi < y)   for xi in x)
    return (greater - lesser) / (n1 * n2)

def pairwise_stats(df, metric, pairs):
    results = []
    for a, b in pairs:
        #xa = df.loc[df['run']==a, metric]
        #xb = df.loc[df['run']==b, metric]
        # Adapt for nans
        group_a = df[(df['run'] == a) & df[metric].notna()][['id', metric]]
        group_b = df[(df['run'] == b) & df[metric].notna()][['id', metric]]
        # Align by shared IDs
        merged = pd.merge(group_a, group_b, on='id', suffixes=('_a', '_b'))
        # Now as usual
        xa = merged[f'{metric}_a'].values
        xb = merged[f'{metric}_b'].values
        stat, p = wilcoxon(xa, xb)
        med_diff = np.median(xa - xb)
        d = cliffs_delta(xa, xb)
        results.append({
            'Model_A': a,
            'Model_B': b,
            'p-value': p,
            'median_diff': med_diff,
            "cliff's delta": d
        })
    return pd.DataFrame(results)


# In[11]:


# Load index data and convert to DataFrames
base_df = recover_index(results_paths["Qwen2.5_BASE"])
grpo_df = recover_index(results_paths["Qwen2.5_GRPO"])
distil_df = recover_index(results_paths["Qwen2.5_distil"])

# Build the big dataframe
for name, df in [
    ("base", base_df),
    ("grpo", grpo_df),
    ("distil", distil_df),
]:
    df["run"] = name
all_df = pd.concat([base_df, grpo_df, distil_df], ignore_index=True)

# Get accuracies
all_df.groupby("run")["correct"].mean()


# In[12]:


# Also get accuracies this way
print("Accuracy for Qwen2.5_BASE:")
print_accuracy(base_df)
print("\nAccuracy for Qwen2.5_GRPO:")
print_accuracy(grpo_df)
print("\nAccuracy for Qwen2.5_distil:") #! Este coger los fallos y comprobarlos a mano, tampoco son tantos
print_accuracy(distil_df)


# In[13]:


# Analyze the correct results
correct_df = all_df[all_df["correct"] == 1].copy()

# Get entropies results
correct_df[["res_layer_entropy", "res_auc_entropy",
            "rea_layer_entropy", "rea_auc_entropy",
            "res_last_entropy", "rea_last_entropy"]] = correct_df.apply(
    lambda r: pd.Series(extract_entropy_data(r, run_paths[r["run"]], tokenizers[r["run"]])),
    axis=1
)

# Get the logp results
correct_df[["res_layer_logp", "res_auc_logp", "res_last_logp"]] = correct_df.apply(
    lambda r: pd.Series(extract_target_avg_logp(r, run_paths[r["run"]])),
    axis=1
)

# Get the forward KL results
correct_df[["res_layer_fkl", "res_auc_fkl",
            "rea_layer_fkl", "rea_auc_fkl",
            "res_last_fkl", "rea_last_fkl"]] = correct_df.apply(
    lambda r: pd.Series(extract_forward_kl_data(r, run_paths[r["run"]], tokenizers[r["run"]])),
    axis=1
)

# Get the top-k final layer summarized results
correct_df = correct_df.join(
    correct_df.apply(lambda r:
        pd.Series(summarise_final_topk(
            load_topk(r, run_paths[r["run"]], "final"),
            r["output_tokens"]                      # or r["answer_token_ids"]
        )), axis=1))


# In[14]:


correct_df.columns


# In[18]:


print("=== res_auc_entropy ===")
print(descriptive_stats(correct_df, 'res_auc_entropy'), "\n")
print("=== rea_auc_entropy ===")
print(descriptive_stats(correct_df, 'rea_auc_entropy'), "\n")
print("=== res_last_entropy ===")
print(descriptive_stats(correct_df, 'res_last_entropy'), "\n")
#print("=== rea_last_entropy ===")
#print(descriptive_stats(correct_df, 'rea_last_entropy'), "\n")

print("=== res_auc_fkl ===")
print(descriptive_stats(correct_df, 'res_auc_fkl'), "\n")
print("=== rea_auc_fkl ===")
print(descriptive_stats(correct_df, 'rea_auc_fkl'), "\n")

#print("=== res_auc_logp ===")
#print(descriptive_stats(correct_df, 'res_auc_logp'), "\n")
print("=== res_last_logp ===")
print(descriptive_stats(correct_df, 'res_last_logp'), "\n")
print("=== final_margin ===")
print(descriptive_stats(correct_df, 'final_margin'), "\n")


# In[19]:


pairs = [('base','grpo'), ('base','distil'), ('grpo','distil')]

print("Pairwise for res_auc_entropy:")
print(pairwise_stats(correct_df, 'res_auc_entropy', pairs), "\n")
print("Pairwise for rea_auc_entropy:")
print(pairwise_stats(correct_df, 'rea_auc_entropy', pairs), "\n")
print("Pairwise for res_last_entropy:")
print(pairwise_stats(correct_df, 'res_last_entropy', pairs), "\n")
#print("Pairwise for rea_last_entropy:")
#print(pairwise_stats(correct_df, 'rea_last_entropy', pairs), "\n")

print("Pairwise for res_auc_fkl:")
print(pairwise_stats(correct_df, 'res_auc_fkl', pairs), "\n")
print("Pairwise for rea_auc_fkl:")
print(pairwise_stats(correct_df, 'rea_auc_fkl', pairs), "\n")

print("Pairwise for res_auc_logp:")
print(pairwise_stats(correct_df, 'res_auc_logp', pairs), "\n")
print("Pairwise for res_last_logp:")
print(pairwise_stats(correct_df, 'res_last_logp', pairs), "\n")
print("Pairwise for final_margin:")
print(pairwise_stats(correct_df, 'final_margin', pairs), "\n")


# **final_margin**
# Es la diferencia de probabilidad entre el rank 1 y el rank2. Cuanto mayor sea, más seguro está el modelo en el token final.
# 

# # Now the plots

# In[21]:


# boxplot
runs = ['base', 'grpo', 'distil']  # order you want on the x-axis
data = [correct_df.loc[correct_df['run']==r, 'final_margin'].values for r in runs]

# ——————————————————————————
# Wada-inspired palette (beniiro, aotake, kigiki)
# ——————————————————————————
colors = ['#1b9e77',  
          '#d95f02',  
          '#7570b3']  

# ——————————————————————————
# Plot
# ——————————————————————————
fig, ax = plt.subplots(figsize=(6, 4))

box = ax.boxplot(
    data,
    labels=["Base", "GRPO", "Distill"],
    notch=False,
    patch_artist=True,
    showfliers=True,
    widths=0.6,
    flierprops=dict(marker='o', markersize=4, markerfacecolor='gray', alpha=0.5, markeredgewidth=0),
    medianprops=dict(color='k', linewidth=1.5),
    whiskerprops=dict(color='k', linewidth=1.2),
    capprops=dict(color='k', linewidth=1.2),
    boxprops=dict(linewidth=1.2)
)

# fill each box with its Wada color
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# axis labels & grid
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Margin', fontsize=12)
ax.set_title('Final Margin by Model', fontsize=14, pad=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

# tighten
plt.tight_layout()
plt.show()


# In[26]:


arrs = np.stack(correct_df[correct_df["run"] == run][col].values)


# In[36]:


correct_df["rea_layer_entropy"]


# In[38]:


# Example DataFrame column names:
# - "run": model name (base, grpo, distil)
# - "layerwise_entropy", "layerwise_forward_kl", "target_avg_logp": lists/arrays per example

# Determine layer count and layer indices
n_layers = len(correct_df["res_layer_entropy"].iloc[0])
layers = np.arange(1, n_layers + 1)  # human‐friendly 1-based indexing

# Color palette (Sanzo Wada inspired)
colors = {
    "base":  "#1b9e77",   # Teal
    "grpo":  "#d95f02",   # Orange
    "distil":"#7570b3"    # Purple
}

# Metrics to plot
metrics = [
    ("res_layer_entropy",    "Entropy",           "Numeric Result Layerwise Entropy"),
    ("rea_layer_entropy",    "Entropy",           "Reasoning Tokens Layerwise Entropy"),
    ("res_layer_fkl",       "Forward KL",        "Numeric Result Layerwise Forward KL"),
    ("rea_layer_fkl",       "Forward KL",        "Reasoning Tokens Layerwise Forward KL"),
    ("res_layer_logp",      "Target Log‐Prob",   "Numeric Result Layerwise Target Log‐Prob"),
]

for col, ylabel, title in metrics:
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in ["base", "grpo", "distil"]:
        # stack per-example trajectories
        #arrs = np.stack(correct_df[correct_df["run"] == run][col].values)
        seqs = correct_df.loc[correct_df["run"] == run, col].dropna().tolist()
        # 2) convert to numpy and keep only the ones with correct length
        valid = []
        for s in seqs:
            arr = np.asarray(s, dtype=float)
            if arr.shape[0] == n_layers:
                valid.append(arr)
        if len(valid) == 0:
            continue
        arrs = np.stack(valid, axis=0)
        
        # compute summary curve and quartile bands
        mean_curve = np.nanmean(arrs, axis=0)
        lower_q25 = np.nanpercentile(arrs, 25, axis=0)
        upper_q75 = np.nanpercentile(arrs, 75, axis=0)
        
        # plot mean trajectory
        ax.plot(layers, mean_curve, label=run.capitalize(),
                color=colors[run], linewidth=2)
        # shade IQR band
        ax.fill_between(layers, lower_q25, upper_q75,
                        color=colors[run], alpha=0.2)

    # Styling
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(layers[::5])
    ax.set_xticklabels(layers[::5])
    ax.set_xlim(1, n_layers)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(title="Model", fontsize=10, title_fontsize=11, loc='right')
    plt.tight_layout()
    plt.show()

