import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
import torch
import re
import argparse
import numpy as np
import random
from tqdm import tqdm
from transformers import (
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

from explainability.models import LLMConfig
from explainability.models import DefaultHFModelLoader, BnBQuantizedModelLoader
from explainability.integrated_gradients import IGConfig
from explainability.integrated_gradients import IGEval
from explainability.integrated_gradients.results_saver import AttributionResultSaver

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

def parse_answer(id, example, ground_truth_answer, model_answer):
    # Process majority voting answers
    numeric_answer = model_answer["numeric"]

    # Check correctness
    correct = (numeric_answer == ground_truth_answer) if numeric_answer is not None else False
    return {
        'id': id,
        'question': example['question'],
        'ground_answer_text': example['answer'],
        'model_answers_text': model_answer['text'],
        'extracted_model_answers': numeric_answer,
        'extracted_ground_answer': ground_truth_answer,
        'majority_answer': numeric_answer,
        'correct': correct,
    }
    
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
    parser.add_argument('--use-majority-vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n-votes', type=int, default=1)
    parser.add_argument("--use-cot-prompt", action="store_true")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--half-eval", action="store_true")
    # IG args:
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--internal-batch-size", type=int, default=10)
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
    
    # Load IG. The IG runs as many as n_steps baselines for integration, separated in batches of internal_batch_size
    ig_config = IGConfig(
        n_steps = args.n_steps,
        internal_batch_size = args.internal_batch_size,
    )
    # Generate all the integrated gradients classes
    ig_target_logprob = IGEval(
        model = model,
        tokenizer = tokenizer,
        forward_func = "target_logprob_forward_func",
        multiply_by_inputs = True,
    )
    ig_target_entropy = IGEval(
        model = model,
        tokenizer = tokenizer,
        forward_func = "target_entropy_forward_func",
        multiply_by_inputs = True,
    )
    ig_seq_logprob = IGEval(
        model = model,
        tokenizer = tokenizer,
        forward_func = "seq_logprob_forward_func",
        multiply_by_inputs = True,
    )
    ig_seq_entropy = IGEval(
        model = model,
        tokenizer = tokenizer,
        forward_func = "seq_entropy_forward_func",
        multiply_by_inputs = True,
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
    
    saver = AttributionResultSaver(model_name='GSM8K_' + args.model.replace('/', '__'),
                        root = 'eval_results/ig')
    
    rows = []

    #results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        # To do just half the test data
        if (i > len(dataset)/2) and (args.half_eval):
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

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
        model_answer = extract_final_answer(outputs, tokenizer)
        
        # tokenize the prompt once up front
        input_ids = tokenizer.encode(input_text, return_tensors="pt")[0]
            
        # Get the output tokens, just like in MATH
        output_tokens = get_model_output(inputs['input_ids'][0], outputs[0], tokenizer)
        # after generation, your `outputs` has shape (1, seq_len)
        seq_len = outputs.size(1)
        
        # Get inputs for the IG
        attention_mask = outputs.new_ones(outputs.shape)  # Create attention mask for the full sequence
        embeds = model.get_input_embeddings()(outputs).detach()   # Get the embeddings for the full sequence
        embeds.requires_grad_()  # Make sure gradients are enabled for the embeddings
        
        # Choose the target position and sequence range
        first_pos = model_answer['answer_token_idxs'][0] if model_answer['answer_token_idxs'] else seq_len - 1
        first_id = model_answer['answer_token_ids'][0] if model_answer['answer_token_ids'] else outputs[0, -1].item()
        sequence_range = model_answer['answer_token_idxs'] if model_answer['answer_token_idxs'] else seq_len - 1
        output_ids = outputs[0].tolist()
        
        # Prepare for averaging
        input_len = inputs['input_ids'].shape[-1]
        output_len = first_pos
        
        # Now get the attributions
        first_target_logprob_attr = ig_target_logprob.target_token_attributions(
            input_embeds = embeds,
            attention_mask = attention_mask,
            target_pos = first_pos,
            predicted_id = first_id,
            ig_config = ig_config,
        )
        # Stats
        avg_input_target_logprob_attr = first_target_logprob_attr[:input_len].mean().item()
        avg_output_target_logprob_attr = first_target_logprob_attr[input_len:output_len].mean().item()
        
        first_target_entropy_attr = ig_target_entropy.target_token_attributions(
            input_embeds = embeds,
            attention_mask = attention_mask,
            target_pos = first_pos,
            predicted_id = first_id,
            ig_config = ig_config,
        )
        # Stats
        avg_input_target_entropy_attr = first_target_entropy_attr[:input_len].mean().item()
        avg_output_target_entropy_attr = first_target_entropy_attr[input_len:output_len].mean().item()
        
        seq_logprob_attr = ig_seq_logprob.seq_token_attributions(
            input_embeds = embeds,
            attention_mask = attention_mask,
            target_positions = sequence_range,
            input_ids = output_ids,
            ig_config = ig_config,
        )
        # Get the average input and output attributions
        avg_input_seq_logprob_attr = seq_logprob_attr[:input_len].mean().item()
        avg_output_seq_logprob_attr = seq_logprob_attr[input_len:output_len].mean().item()
        
        seq_entropy_attr = ig_seq_entropy.seq_token_attributions(
            input_embeds = embeds,
            attention_mask = attention_mask,
            target_positions = sequence_range,
            input_ids = output_ids,
            ig_config = ig_config,
        )        
        # Get the average input and output attributions
        avg_input_seq_entropy_attr = seq_entropy_attr[:input_len].mean().item()
        avg_output_seq_entropy_attr = seq_entropy_attr[input_len:output_len].mean().item()
        
        # Free the GPU memory
        first_target_logprob_attr = first_target_logprob_attr.detach().cpu()
        first_target_entropy_attr = first_target_entropy_attr.detach().cpu()
        seq_logprob_attr = seq_logprob_attr.detach().cpu()
        seq_entropy_attr = seq_entropy_attr.detach().cpu()
        
        # Get if correct, I reuse old function
        parsed_answer = parse_answer(
            i, example, ground_truth_answer, model_answer
        )
        
        # Things to consider saving
        correct = parsed_answer['correct']
        question = input_text
        output_tokens = output_tokens.tolist()
        qid = str(i).zfill(6).split('/')[-1].replace('.json','')
        
        # ... for each example in your QA eval loop:
        qid = i
        question = example['question']

        # 2) Save
        saver.add(
            qid=qid,
            question=input_text,    # Pongo la input prompt instead of question text
            correct = correct,
            numeric_answer = model_answer["numeric"],
            true_answer = ground_truth_answer,
            input_tokens=input_ids.tolist(),    # Don't give tensor to save in json
            output_tokens=output_tokens,
            # target logprob
            first_target_logprob_attr=first_target_logprob_attr,
            avg_input_target_logprob_attr=avg_input_target_logprob_attr,
            avg_output_target_logprob_attr=avg_output_target_logprob_attr,
            # target entropy
            first_target_entropy_attr=first_target_entropy_attr,
            avg_input_target_entropy_attr=avg_input_target_entropy_attr,
            avg_output_target_entropy_attr=avg_output_target_entropy_attr,
            # sequence logprob
            seq_logprob_attr=seq_logprob_attr,
            avg_input_seq_logprob_attr=avg_input_seq_logprob_attr,
            avg_output_seq_logprob_attr=avg_output_seq_logprob_attr,
            # sequence entropy
            seq_entropy_attr=seq_entropy_attr,
            avg_input_seq_entropy_attr=avg_input_seq_entropy_attr,
            avg_output_seq_entropy_attr=avg_output_seq_entropy_attr,
            # any extra blobs, e.g. raw_attr=raw_attr_tensor
        )
        
        # Delete leftovers from GPU
        del embeds, outputs, attention_mask
        torch.cuda.empty_cache()
        
            
    saver.close()
    
if __name__ == '__main__':
    main()

