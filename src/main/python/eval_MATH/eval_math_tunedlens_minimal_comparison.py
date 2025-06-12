import io
import logging
import math
import os
import pprint
import re
import sys
import json
import time
import transformers
import numpy as np
import psutil

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.MATH import MATHDataset
from dataset.khan_academy import KhanAcademyMathDataset
from dataset.util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv

from explainability.models import LLMConfig
from explainability.models import DefaultHFModelLoader, BnBQuantizedModelLoader
from explainability.tuned_lens import TunedLens

from eval_MATH.results_saver import ResultSaver

from collections import OrderedDict


BOXED_PATTERN = re.compile(r'\\boxed\s*{')

def log_resource_usage(label=""):
    """
    Debugging function to log resource usage (CPU and RAM).
    """
    process = psutil.Process(os.getpid())
    # CPU percent since last call; use a small interval to get an updated snapshot
    cpu_pct = process.cpu_percent(interval=0.1)
    # Resident Set Size (RSS) in bytes → convert to GB
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"[RESOURCE] {label} | CPU%: {cpu_pct:.2f}% | RAM: {mem_gb:.2f} GB")

def get_level_type(fname):
    """
    Somewhat inefficient, but much easier than changing dataloader and probably fine for evaluation
    """
    with open(fname, 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e
    level, prob_type = problem_data['level'], problem_data['type']
    try:
        level = int(level.split("Level ")[1])
    except:
        level = None
    return level, prob_type

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def dict_to_gpu(d, device_id=None):
    new_dict = dict()
    for key, value in d.items():
        # Only move to GPU is cuda() is a function
        if 'cuda' in dir(value):
            new_dict[key] = value.cuda(device_id)
        else:
            new_dict[key] = value
    return new_dict

def _find_box_char_span(text: str):
    """
    Return (char_start, char_end) indices (inclusive)
    of the expression inside the *first* \boxed{ … } in `text`.
    If none is found, return (None, None).
    """
    m = BOXED_PATTERN.search(text)  # Gets the first occurence of \boxed{
    if m is None:
        return None, None                      # no \boxed{ … } at all

    start = m.end()            # Guardamos el start, este ya está claro, es el siguiente elemento al \\boxed{}
    depth = 1                  # brace nesting level

    for idx in range(start, len(text)):
        if text[idx] == '{':
            depth += 1      # Si abro corchete, me meto en un nesting adicional
        elif text[idx] == '}':
            depth -= 1      # Si cierro corchete, salgo de un nesting
            if depth == 0:      # matched the opening '{' of \boxed{
                return start, idx        # inclusive on the right

    return None, None           # unbalanced brace – treat as not found

def get_real_sol_idxs(tokens_sol: torch.Tensor, tokenizer):
    """
    Locate the token span (start_idx, stop_idx) of the content inside the
    *first* LaTeX \\boxed{ … } in `tokens_sol`.
    
    *Note*: Consigo sacar el intervalo en tokens que contiene al intervalo en str
    Parameters
    ----------
    tokens_sol : torch.Tensor   shape (1, L) – *just the generated part*
    tokenizer  : HF tokenizer   – must implement encode / decode

    Returns
    -------
    (int | None, int | None)    – start and end indices **w.r.t. tokens_sol**
    """
    # 1. Decode once – *without* dropping special tokens
    decoded = tokenizer.decode(tokens_sol, skip_special_tokens=False)

    # 2. Find the character span of the answer
    char_l, char_r = _find_box_char_span(decoded)
    if char_l is None:
        return None, None                     # no boxed answer detected

    ## 3. How many tokens appear *before* the answer?
    ##    (encode with add_special_tokens=False to stay compatible with any model)
    #prefix_ids  = tokenizer.encode(
    #    decoded[:char_l], add_special_tokens=False)
    #inside_ids  = tokenizer.encode(
    #    decoded[char_l:char_r+1], add_special_tokens=False)
#
    #left_idx  = len(prefix_ids)     # Ojo que es el primer token que NO incluye nada de \\boxed{, pues eso sería verdaderamente el comienzo de la solución
    #right_idx = left_idx + len(inside_ids) - 1   # inclusive. No será -2???

    char_l_incl = char_l - 1

    # 3) Re-tokenize *with* offset mappings
    enc = tokenizer(
        decoded,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    offsets = enc["offset_mapping"]   # list of (char_start, char_end) for each token

    # 4) Find which token contains char_l and which contains char_r
    left_idx = next((i for i,(s,e) in enumerate(offsets) if s <= char_l_incl < e), None)
    right_idx = next((i for i,(s,e) in enumerate(offsets) if s <= char_r      < e), None)

    return left_idx, right_idx+1

def get_topk_final_log_probs(
        output_tokens: torch.Tensor,
        log_probs:      torch.Tensor,    # (n_layers , seq_len , vocab)
        tokenizer,
        topk: int = 5
):
    """
    Return 3 items (all align on dimension-0 == seq_len):

      • topk_log_probs : torch.FloatTensor (seq_len , K)
      • topk_ids       : torch.LongTensor  (seq_len , K)
      • topk_tokens    : List[List[str]]   len == seq_len, inner len == K

    The *actual* generated token for each position is guaranteed to be
    in the top-K set (last slot if needed).
    """
    ids        = output_tokens.tolist()          # (seq_len,)
    seq_len    = log_probs.shape[1]
    # log_probs was turned to numpy when parsing lp, so we need to go back to torch tensor
    if isinstance(log_probs, np.ndarray):
        log_probs = torch.from_numpy(log_probs)
    last_layer = log_probs[-1]                      # (seq_len , vocab)

    topk_logp_rows   = []
    topk_id_rows     = []
    topk_token_rows  = []

    for pos in range(seq_len):
        lp_row = last_layer[pos]                 # (vocab,)
        # 1) get native top-K
        vals, idxs = lp_row.topk(topk)           # tensors, size K
        top_ids    = idxs.tolist()
        top_vals   = vals.tolist()

        # 2) make sure the real token is included
        actual_id   = ids[pos]
        actual_lp   = lp_row[actual_id].item()
        if actual_id not in top_ids:
            top_ids[-1]  = actual_id
            top_vals[-1] = actual_lp

        # 3) decode → tokens
        tok_strs = tokenizer.convert_ids_to_tokens(top_ids)

        # 4) append row
        topk_logp_rows.append(top_vals)
        topk_id_rows.append(top_ids)
        topk_token_rows.append(tok_strs)

    topk_log_probs = torch.tensor(topk_logp_rows, dtype=torch.float32)
    topk_ids       = torch.tensor(topk_id_rows,   dtype=torch.long)
    topk_tokens    = topk_token_rows              # already Python lists

    return topk_log_probs, topk_ids, topk_tokens


def run_eval(args):
    
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    
    print('Loading base model and tokenizer...')    # Denote base model and things with 
    model_config = LLMConfig(
        model_name = args.model,
        #device_map = args.device_map,
        torch_dtype = torch.float16,
    )
    
    # Get HF token
    hf_key = os.getenv("HF_TOKEN")
    if hf_key is None:
        if args.quantized:
            model_loader = BnBQuantizedModelLoader(model_config)
        else:
            model_loader = DefaultHFModelLoader(model_config)
    else:
        if args.quantized:
            model_loader = BnBQuantizedModelLoader(model_config, hf_key = hf_key)
        else:
            model_loader = DefaultHFModelLoader(model_config, hf_key = hf_key)
    model, tokenizer = model_loader.load_model()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    print('\nLoading tuned lens...')
    assert args.tuned_lens is not None, "Please provide a tuned lens model path."
    tuned_lens = TunedLens(
        model = model,
        tokenizer = tokenizer,
        lens = args.tuned_lens,
    )

    print('\nLoading dataset...')
    eval_data = get_dataset(args)
    for inner_dset in eval_data.datasets:
        inner_dset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=1, 
        num_workers=0, 
        pin_memory=True,
    )
    
    saver = ResultSaver(model_name='MATH_' + args.model.replace('/', '__'))

    """
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(dataloader)):
            batch = dict_to_gpu(batch, device_id=0)
            print(batch['fnames'])
            print(batch['input_ids'])
            quit()
    """
    
    model = model.eval().cuda()
    tuned_lens.tuned_lens = tuned_lens.tuned_lens.cuda()

    loss_moving_average = 0

    outputs = []
    answers = []
    types = []
    levels = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}

    with torch.no_grad():
        correct = 0
        total = 0
        skipped = 0
        mean_max_probs_correct = []
        mean_max_probs_wrong   = []
        for i, batch in enumerate(tqdm(dataloader)):
            
            log_resource_usage(f"before example {i}")

            if torch.sum(batch['input_ids']) == 0:
                skipped += 1
                print("SKIPPING", batch['fnames'][0])
                continue

            fnames = batch['fnames'][0]
            assert len(fnames) == 1
            fnames_list.append(fnames[0])
            prob_level, prob_type = get_level_type(fnames[0])
            batch = dict_to_gpu(batch, device_id=0)
            
            # Get attention mask
            attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).long()
            attention_mask = attention_mask.to(batch["input_ids"].device)

            output_ids = model.generate(
                batch['input_ids'], 
                attention_mask=attention_mask,
                eos_token_id   = tokenizer.eos_token_id,
                pad_token_id   = tokenizer.pad_token_id,
                num_beams=args.num_beams, 
                early_stopping=args.early_stopping,
                temperature=1.0,
                max_new_tokens=args.max_new_tokens,
            )
            
            log_resource_usage(f"After generation {i}")

            mean_probs_sol = 0  # Esto realmente no lo estoy usando pero no quiero borrar demasiado código
            
            # Get the output tokens, without context
            output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)
            # Use it to get the boxed solution
            real_sol_start_idx, real_sol_stop_idx = get_real_sol_idxs(output_tokens, tokenizer)
            if real_sol_start_idx is None or real_sol_stop_idx is None:
                skipped += 1
                print("BAD ANSWER, SKIPPING", batch['fnames'][0])
                # Igual mejor contar la skipped como que está mal, Porque estoy haciendo los 300 primeros y si hago 
                # skip sin sumar al total, no cuenta hacia ese número e igual sí que cuentan otras preguntas más sencillas
                equiv = False
                total += 1
                continue    #! No había pensado en esto, está bien skippear cuando no encuentras

            # Print this iteration
            output_str = tokenizer.decode(output_tokens)
            output_full = output_str
            output_str = last_boxed_only_string(output_str)

            if args.math_mode == "eval_peeking":
                answer_str = last_boxed_only_string(tokenizer.decode(batch['labels'][0]))
            else:
                answer_str = tokenizer.decode(batch['labels'][0])

            output, answer = remove_boxed(output_str), remove_boxed(answer_str)
            
            # Get ground truth answer
            ground_truth = batch['labels'][0]
            
            # Pass through tuned lens
            input_len = len(batch['input_ids'][0])
            seq_len = output_ids.shape[1]    # Assuming output_tokens contains question too # TODO DEBUG
            if args.last_tokens_tl is not None: # For R1 case evaluation is too slow for long context
                start_len = max(seq_len - args.last_tokens_tl, input_len)   # just in case seq_len is smaller than last_tokens_tl
                token_range = (start_len, seq_len)
            else:
                token_range = (input_len, seq_len)
            
            #! test
            log_resource_usage(f"Before minimal stats_explain, example {i}")

            minimal_out = tuned_lens.stats_explain_minimal(
                input_ids  = output_ids[0].tolist(),
                token_range= token_range,
                layer_stride = 1,   # or 2, or 4—whatever you decide
            )
            
            min_lp_all       = minimal_out["log_probs"]    # shape (num_layers, slice_len, vocab_size)
            min_entropy_all  = minimal_out["entropy"]      # shape (num_layers, slice_len)
            min_forwardkl_all= minimal_out["forward_kl"]   # shape (num_layers, slice_len)

            log_resource_usage(f"After minimal stats_explain, example {i}")
            
            
            
            #log_resource_usage(f"before tunedlens {i}")
            
            orig_traj = tuned_lens.stats_explain(
                input_ids = output_ids[0].tolist(),
                statistic = ["entropy", "cross_entropy", "forward_kl"],
                layer_stride=1,
                token_range = token_range,
            )
            
            log_resource_usage(f"after tunedlens {i}")
            
            # orig_lp_all: NumPy array of shape (num_layers, slice_len, vocab_size)
            orig_lp_all = orig_traj.log_probs.cpu().numpy() if isinstance(orig_traj.log_probs, torch.Tensor) \
                        else orig_traj.log_probs
            # orig_entropy_all: NumPy of shape (num_layers, slice_len)
            orig_entropy_all = orig_traj.entropy().stats
            # orig_forwardkl_all: NumPy of shape (num_layers, slice_len)
            orig_forwardkl_all = orig_traj.forward_kl().stats
            
            
            #  3) Compare shapes first:
            assert orig_lp_all.shape       == min_lp_all.shape,       f"log_probs shapes differ: {orig_lp_all.shape} vs {min_lp_all.shape}"
            assert orig_entropy_all.shape  == min_entropy_all.shape,  f"entropy shapes differ: {orig_entropy_all.shape} vs {min_entropy_all.shape}"
            assert orig_forwardkl_all.shape== min_forwardkl_all.shape,f"forward_kl shapes differ: {orig_forwardkl_all.shape} vs {min_forwardkl_all.shape}"

            #  4) Now compare values (allowing for tiny floating‐point differences):
            lp_equal       = np.allclose(orig_lp_all,       min_lp_all,       rtol=1e-5, atol=1e-6)
            entropy_equal  = np.allclose(orig_entropy_all,  min_entropy_all,  rtol=1e-5, atol=1e-6)
            forwardkl_equal= np.allclose(orig_forwardkl_all,min_forwardkl_all,rtol=1e-5, atol=1e-6)

            print("log_probs match:",        lp_equal)
            print("entropy match:",          entropy_equal)
            print("forward_kl match:",       forwardkl_equal)

            #  5) If you see `True` for all three, they are numerically identical.
            #    If any is False, you can print the maximum absolute difference:
            if not lp_equal:
                diff = np.abs(orig_lp_all - min_lp_all).max()
                print("  max |Δ log_probs| =", diff)
            if not entropy_equal:
                diff = np.abs(orig_entropy_all - min_entropy_all).max()
                print("  max |Δ entropy| =", diff)
            if not forwardkl_equal:
                diff = np.abs(orig_forwardkl_all - min_forwardkl_all).max()
                print("  max |Δ forward_kl| =", diff)
            
            
            
            total += 1
            
            if args.max_examples is not None and total >= args.max_examples:
                break

    saver.close()
    
    subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']

    print(f"Average of mean_max_probs_correct = {sum(mean_max_probs_correct)}/{len(mean_max_probs_correct)} = ", sum(mean_max_probs_correct)/len(mean_max_probs_correct))
    print(f"Average of mean_max_probs_wrong   = {sum(mean_max_probs_wrong)}/{len(mean_max_probs_wrong)} = ", sum(mean_max_probs_wrong)/len(mean_max_probs_wrong))

    # now save outputs and answers
    with open(f"outputs_answers_Temp2e-1_{args.model.replace('/', '__')}.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        # print(cors)
        for prob_type in subjects:
            for prob_level in [1, 2, 3, 4, 5]:
                if (prob_level, prob_type) in cors:
                    cors_list = cors[(prob_level, prob_type)]
                    print("{} Level {} Accuracy = {}/{} = {:.3f}".format(prob_type, prob_level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                    f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(prob_type, prob_level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))

        print("#####################")
        f.write("#####################\n")
        # also get accuracies for each 
        for level in sorted(level_cors):
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")

        for subject in subjects:
            # for subject in sorted(subject_cors):
            if subject in subject_cors:
                cors_list = subject_cors[subject]
                print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        print("Skipped = {}".format(skipped))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))
        f.write("Skipped = {}".format(skipped))
    
    print()
    
def get_model_output(context, full_output, tokenizer):
    """
    Given the context and the full model output (context + generated),
    extract just the generated tokens.
    Remove the last token if it is <|endoftext|>
    """
    ret = full_output[len(context):]
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret

def get_dataset(args):
    all_datasets = []

    if args.math_dataroot is not None:
        if args.math_mode == 'gpt2-eval':
            all_datasets.append(
                MATHDataset(
                    dataroot=args.math_dataroot, 
                    tokenizer=None, # Set in run_training(), not in dataset creation 
                    max_tokens=args.max_new_tokens,
                    mode='gpt2-eval', 
                )
            )
        else:
            all_datasets.append(
                MATHDataset(
                    dataroot=args.math_dataroot, 
                    tokenizer=None, # Set in run_training(), not in dataset creation 
                    max_tokens=args.max_new_tokens,
                    mode='gpt2-eval',
                    mode_answer=args.math_mode,
                    peek_fraction=args.peek_fraction
                )
            )

    
    train_data = torch.utils.data.ConcatDataset(all_datasets)
    return train_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-14B')
    parser.add_argument('--tuned-lens', type=str, default=None)
    parser.add_argument('--num-beams', default=1, type=int) # Kept at None
    parser.add_argument('--early-stopping', default=False, action='store_true') # Only for beam generation
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--tokenizer-merges-file', default=None, type=str)
    parser.add_argument("--output-folder", type=str, default="data/eval_results/single_eval/")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    # Dataloading
    parser.add_argument('--math-dataroot', default="MATH/test/*/*.json", type=str)
    parser.add_argument('--math-mode', default='gpt2-eval', type=str)
    parser.add_argument('--peek-fraction', type=float, default=1.0)
    parser.add_argument('--max-examples', type=int, default=None)
    
    # Others
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--last-tokens-tl', default = None, type = int)
    parser.add_argument('--quantized', action='store_true', default=False, help='Use quantized model')

    args = parser.parse_args()

    run_eval(args)
