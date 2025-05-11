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
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--tuned_lens', type=str, default=None)
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    parser.add_argument("--use_cot_prompt", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--output_folder", type=str, default="data/eval_results/single_eval/")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--half_eval", action="store_true")
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
    model_loader = DefaultHFModelLoader(model_config, hf_key = "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")
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
    
    run_dir = Path(args.output_folder) / f"{args.model.split('/')[-1]}_{datetime.now():%Y%m%dT%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir/"meta.json","w"), indent=2, default=str)
    
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
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=args.max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
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
            # after generation, your `outputs` has shape (1, seq_len)
            seq_len = outputs.size(1)
            # the newly generated tokens are at positions [input_len, seq_len)
            token_range = (input_len, seq_len)
            lens_output = tuned_lens.stats_explain(
                prompt = answer['full_text'],    # answer no es, tiene que ser con el input
                statistic = ["entropy", "log_probs"],
                layer_stride = 1,
                token_range = token_range,
            )
            
            # Get ground truth answer logprob throughout the layers
            log_probs = lens_output.log_probs
            ground_enc_ids = tokenizer(ground_truth_answer, add_special_tokens = False)['input_ids']
            
            
            
            
            # TODO DEBUG
            #answer['input'] = input_text
            #answer['ground_truth_answer'] = ground_truth_answer
            #answer['input_len'] = input_len
            #json_path = "../data/eval_results/particular_cases/20250509_R1_answer.json"
            #json.dump(answer, open(json_path, "w"), indent=2, default=str)
            #path = "../data/eval_results/particular_cases/20250509_R1_tensors.npz"
            #log_probs_out = log_probs.astype(np.float16)
            #entropy_out = lens_output.entropy().stats.astype(np.float16)
            #np.savez_compressed(path,
            #                log_probs = log_probs_out,
            #                entropy = entropy_out)
            # TODO
            
            
            
            # Now get the log probs for each cipher in the numeric answer
            ground_log_probs = []
            if 'DeepSeek-R1' in args.model:
                offset = input_len
            else:
                offset = input_len + 1
            if (answer['answer_token_ids'] is not None) and (ground_enc_ids is not None):   # Fix nonetype error when creating zip
                for idx, ground_id in zip(answer['answer_token_idxs'], ground_enc_ids): # zip takes the shortest vec length
                    ground_log_probs.append(log_probs[:, idx - offset, ground_id])   # el -input_len-1 ha sido checked manualmente
            else:
                ground_log_probs = None
            # Now stack and mean (if not empty)
            if ground_log_probs:
                stacked_lp          = np.stack(ground_log_probs, axis=1)   # (num_layers, n_positions)
                layer_avg_logprob   = stacked_lp.mean(axis=1)                 # (num_layers,)
            else:
                # fall‑back if something went weird (shouldn’t normally happen)
                layer_avg_logprob   = np.full((log_probs.shape[0],), float('nan'))
            
            # Get entropy for the answer part of the text
            output_entropy = lens_output.entropy().stats # This already doesn't include input, as token range was as input for tuned lens
            
            # Store in dict
            lens_answers_dict = {
                'log_probs': layer_avg_logprob,
                'entropy': output_entropy,
            }
            lens_answers.append(lens_answers_dict)
        
        #! save lens_answers? big np.arrays     
        tensor_paths = save_tensors(i, lens_answers, run_dir)
        rows.append(parse_answer(i, example, ground_truth_answer,
                                model_answers) |        # PEP 584 merge
                    {"tensor_path": tensor_paths[0]})
        
        #results.append(parse_answer(i, example, ground_truth_answer, model_answers, lens_answers))
        
        #break #! DEBUGGING PURPOSES
    
    pd.DataFrame(rows).to_parquet(run_dir/"results.parquet", index=False)
    print(f"Saved tensors to {run_dir}")
    print(f"Accuracy = {pd.read_parquet(run_dir/'results.parquet')['correct'].mean():.4f}")
    
    #dump_single_results(args, results)
    
if __name__ == '__main__':
    main()

