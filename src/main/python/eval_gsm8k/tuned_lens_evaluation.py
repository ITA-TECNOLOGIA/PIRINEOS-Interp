import torch
import re
import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList
)
from pathlib import Path
import pandas as pd
from eval_gsm8k.utils import (
    SpecificStringStoppingCriteria,
    extract_predicted_answer,
    extract_ground_truth
)
from eval_gsm8k.results_saver import ResultSaver
from datasets import load_dataset
from collections import Counter
import json
from datetime import datetime

from explainability.models import LLMConfig
from explainability.models import DefaultHFModelLoader, BnBQuantizedModelLoader
from explainability.tuned_lens import TunedLens


def save_tensors(sample_id: int,
                 lens_answers: list[dict[str, np.ndarray]],
                 run_dir: Path) -> list[str]:
    """
    Store each answer's arrays in <run_dir>/tensors and return their paths.
    """
    (run_dir / "tensors").mkdir(parents=True, exist_ok=True)
    tensor_paths = []
    for j, ans in enumerate(lens_answers):
        path = run_dir / "tensors" / f"sample_{sample_id:06d}_{j}.npz"
        # cast to float32 or even float16 if space is a concern
        np.savez_compressed(path,
                            log_probs=ans["log_probs"].astype(np.float32),
                            entropy  =ans["entropy"].astype(np.float32))
        tensor_paths.append(str(path))
    return tensor_paths

def extract_final_answer(outputs, tokenizer):
    # Decode the output tokens
    ids = outputs[0].tolist()
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    # Get the answer section and the numeric answer
    answer_section = decoded.split("A:")[-1].strip() 
    numeric_answer = extract_predicted_answer(answer_section)
    
    # Log the results we have by now
    result_dict = {
        'text': answer_section, 
        'numeric': numeric_answer, 
        'full_text': decoded, 
        'seq_len': outputs.size(1),
        "answer_tokens": None,
        "answer_token_ids": None,
        "answer_token_idxs": None,
    }
    if numeric_answer is None:
        return result_dict

    # Locate the answer position in the string
    start_char = decoded.rfind(numeric_answer)
    # of course if can't find the answer, just return the preliminar result
    if start_char == -1:
        return result_dict
    end_char = start_char + len(numeric_answer)
    
    # Now first get again the token ids but with the offset mapping
    enc = tokenizer(decoded, add_special_tokens = False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    all_ids = enc["input_ids"]
    all_tokens = tokenizer.convert_ids_to_tokens(all_ids)
    
    # Now get what tokens are in the numeric answer (may be multiple as 100 is 3 tokens for example)
    idxs = [
        i for i, (start, end) in enumerate(offsets)
        if not (end <= start_char or start >= end_char)
    ]
    # Get the results
    result_dict["answer_tokens"] = [all_tokens[i] for i in idxs]
    result_dict["answer_token_ids"] = [all_ids[i] for i in idxs]
    result_dict["answer_token_idxs"] = idxs
    return result_dict

def stats_from_layers(lens_output):
    # Get the statistics from the tuned lens output
    stats = {
        'entropy': lens_output.entropy(),
        'log_probs': lens_output.log_probs,
        # Add other statistics as needed
    }
    return stats

def parse_answer(id, example, ground_truth_answer, model_answers):
    # Process majority voting answers
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered_answers = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

    # Check correctness
    correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
    return {
        'id': id,
        'question': example['question'],
        'ground_answer_text': example['answer'],
        'model_answers_text': [ma['text'] for ma in model_answers],
        'extracted_model_answers': numeric_answers,
        'extracted_ground_answer': ground_truth_answer,
        'majority_answer': majority_answer,
        'correct': correct,
    }
    
def dump_single_results(args, results):
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")
    
    results.append({'accuracy': cnt / total})

    os.makedirs(args.output_folder, exist_ok=True)
    model_name = args.model.split('/')[-1]
    
    result_file = os.path.join(args.output_folder, model_name)
    if args.use_cot_prompt:
        result_file += "_cot"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file += f"_results_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
    
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
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-14B')
    parser.add_argument('--tuned_lens', type=str, default=None)
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--output_folder", type=str, default="data/eval_results/single_eval/")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--half_eval", action="store_true")
    # New args
    parser.add_argument('--last-tokens-tl', default = None, type = int)
    parser.add_argument('--quantized', action='store_true', help='Use quantized model with BnB')
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading base model and tokenizer...')    # Denote base model and things with 
    model_config = LLMConfig(
        model_name = args.model,
        device_map = args.device_map,
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
    
    print('\nLoading tuned lens...')
    assert args.tuned_lens is not None, "Please provide a tuned lens model path."
    tuned_lens = TunedLens(
        model = model,
        tokenizer = tokenizer,
        lens = args.tuned_lens,
    )
    
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')   #! Great! There is a test split
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>",
        # The following tokens are specific for Qwen models https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py
        "<|endoftext|>",
        "Question:",
    ]
    
    saver = ResultSaver(model_name='GSM8K_' + args.model.replace('/', '__'))
    
    rows = []

    #results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        # To do just half the test data
        if (i > len(dataset)/4) and (args.half_eval):
            break
        
        example = dataset[i]
        if args.use_cot_prompt:
            input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
        else:
            input_text = 'Q: ' + example['question'] + '\nA:'
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        # Define a stopping condition for generation
        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
                model_answers.append(extract_final_answer(outputs, tokenizer))
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
            model_answers.append(extract_final_answer(outputs, tokenizer))
            
            
        # Pass through tuned lens
        # tokenize the prompt once up front
        input_ids = tokenizer.encode(input_text, return_tensors="pt")[0]
        input_len = input_ids.size(0)
        lens_answers = []
        for answer in model_answers:
            # Get the output tokens, just like in MATH
            output_tokens = get_model_output(inputs['input_ids'][0], outputs[0], tokenizer)
            # Print this iteration
            output_str = tokenizer.decode(output_tokens)
            # after generation, your `outputs` has shape (1, seq_len)
            seq_len = outputs.size(1)
            # the newly generated tokens are at positions [input_len, seq_len)
            if args.last_tokens_tl is not None: # For R1 case evaluation is too slow for long context
                start_len = max(seq_len - args.last_tokens_tl, input_len)   # just in case seq_len is smaller than last_tokens_tl
                token_range = (start_len, seq_len)
            else:
                token_range = (input_len, seq_len)

            lens_output = tuned_lens.stats_explain_minimal(
                input_ids = outputs[0].tolist(),    # answer no es, tiene que ser con el input
                layer_stride = 1,
                token_range = token_range,   # or 2, or 4—whatever you decide
            )
            
            # Parse output
            lp = lens_output["log_probs"]
            entropy = lens_output["entropy"]
            forward_kl = lens_output["forward_kl"]
            
            # Get log probs of the solution
            ground_enc_ids = tokenizer(ground_truth_answer, add_special_tokens = False)['input_ids']   
            
            # Now get the log probs for each cipher in the numeric answer
            ground_log_probs = []
            if 'DeepSeek-R1' in args.model:
                offset = input_len
            else:
                offset = input_len + 1
            if (answer['answer_token_ids'] is not None) and (ground_enc_ids is not None):   # Fix nonetype error when creating zip
                for idx, ground_id in zip(answer['answer_token_idxs'], ground_enc_ids): # zip takes the shortest vec length
                    ground_log_probs.append(lp[:, idx - offset, ground_id])   # el -input_len-1 ha sido checked manualmente
            else:
                ground_log_probs = None
                
            # Now stack and mean (if not empty)
            if ground_log_probs:
                stacked_lp          = np.stack(ground_log_probs, axis=1)   # (num_layers, n_positions)
                layer_avg_logprob   = stacked_lp.mean(axis=1)                 # (num_layers,)
            else:
                # fall‑back if something went weird (shouldn’t normally happen)
                layer_avg_logprob   = np.full((lp.shape[0],), float('nan'))
            
            # Get topk final token log probs
            topk_offset = 1 # I'm using output_ids and token_range so no need to do anything about last_tokens_lp
            topk_final_log_probs, topk_final_ids, topk_final_tokens = get_topk_final_log_probs(
                outputs[0][token_range[0]+topk_offset:token_range[1]], lp[:,:-topk_offset,:], tokenizer, topk=10 # Watch out, they don't match otherwise
            )    
            
            # Get topk log probs by layer # log_probs -> (layers, seq_len, vocab)
            topk_logp, topk_idx  = torch.from_numpy(lp).topk(k=10, dim=-1)      # same shape but …,topk
            flat                 = tokenizer.batch_decode(topk_idx.flatten())
            topk_tokens          = np.array(flat, dtype=object).reshape(topk_idx.shape)
            
            # Get if correct, I reuse old function
            parsed_answer = parse_answer(
                i, example, ground_truth_answer, model_answers
            )
            
            
            # after you have everything computed:
            saver.add(
                qid               = str(i).zfill(6).split('/')[-1].replace('.json',''),
                question          = input_text,
                ground_answer_text= ground_truth_answer,
                model_answers_text= [output_str],
                extracted_model_answers=[answer["numeric"]],
                extracted_ground_answer=ground_enc_ids,
                majority_answer   = answer["numeric"],
                correct           = parsed_answer['correct'],
                output_tokens     = output_tokens.tolist(),

                entropy           = entropy,       # fixed blob
                forward_kl        = forward_kl,
                layer_avg         = torch.tensor(layer_avg_logprob),

                # ─── free-form extras – EACH becomes its own .pt file ───
                topk_final_log_probs = topk_final_log_probs,
                topk_final_ids       = topk_final_ids,
                topk_final_tokens    = topk_final_tokens,

                # same pattern for “layer-wise” info if you want it separate
                topk_layer_log_probs = topk_logp.cpu(),
                topk_layer_ids       = topk_idx.cpu(),
                topk_layer_tokens    = topk_tokens,
            )
            
    saver.close()
    
if __name__ == '__main__':
    main()

