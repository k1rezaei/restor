from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
import math

import torch
import gc
torch.cuda.empty_cache()

from eval_utils import evaluate_model_on_fact_dataset

CLEAN_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
LORA_PATH = ''
PATH_TO_DSET = ''
UNLEARNED_MODELS_PATH = ''
LIMA_LORA_PATH = ''

def save(list_of_log_probs: list, dir_to_save: str):
    agg = []
    for log_prob in list_of_log_probs:
        agg.extend(list(log_prob))
    agg = np.array(agg)
    np.save(dir_to_save, agg)


@torch.no_grad()
def evaluate_model_next_token_pred(model, tokenizer, device, dataset, seed, dir_to_save):
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()

    log_next_token_pred = []
    ppx, tot_len, tot_tokens = 0, 0, 0

    for sample in tqdm(dataset):
        prompt = sample['text']

        tot_len += len(prompt)
        inputs_tokenized = tokenizer([prompt], return_tensors="pt", padding=False).to(device)
        labels = inputs_tokenized.input_ids.to(device)

        inputs_tokenized = {key: value.to(device) for key, value in inputs_tokenized.items()}
        tot_tokens += len(inputs_tokenized['input_ids'][0])
        outputs = model(**inputs_tokenized, labels=labels)

        logits = outputs.logits
        loss = outputs.loss.item()

        sanity = 0
        curr_logits = logits[0, 0: -1]

        log_probs = torch.log_softmax(curr_logits, dim=-1)
        answer_token_ids = inputs_tokenized['input_ids'][0, 1:]
        answer_log_probs = log_probs[range(len(inputs_tokenized['input_ids'][0]) - 1), answer_token_ids]
        sanity += answer_log_probs.sum().item()
        num_of_tokens = (labels != -100).sum().item()
        log_next_token_pred.append(answer_log_probs.cpu().numpy())

        assert (sanity / (num_of_tokens - 1) + loss) ** 2 < 0.1

        ppx += math.exp(loss)

    ppx /= len(dataset)    
    tot_tokens /= len(dataset)
    tot_len /= len(dataset)

    return log_next_token_pred, ppx, tot_tokens, tot_len




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dset', type=str, default='')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--corrupted_model_path', type=str,)
    parser.add_argument('--corrupted_model_name', type=str,)
    parser.add_argument('--corrupted_dataset', type=str, default='')
    parser.add_argument('--unlearning_configs', type=str, default='')
    parser.add_argument('--corrupted_model_config', type=str, default='')
    parser.add_argument('--lima_config', type=str, default='')
    parser.add_argument('--status', type=int, default=0)


    args = parser.parse_args()

    
    if args.status == 0:
        print('corrupted model ...')
        base_model = LlamaForCausalLM.from_pretrained(CLEAN_MODEL_NAME)
        lora_path = f'{LORA_PATH}/{args.corrupted_model_path}'
        model = PeftModel.from_pretrained(base_model, lora_path)
        tokenizer = AutoTokenizer.from_pretrained(lora_path, padding_side='left')
    elif args.status == 1:
        print('clean model ...')
        model = LlamaForCausalLM.from_pretrained(CLEAN_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(CLEAN_MODEL_NAME, padding_side='left')
    else:
        print("lima ft unlearned model")
        print(f'LORA PATH: {LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}')

        base_model = LlamaForCausalLM.from_pretrained(f'{UNLEARNED_MODELS_PATH}/{args.corrupted_dataset}/{args.unlearning_configs}/{args.corrupted_model_config}/checkpoints')
        model = PeftModel.from_pretrained(base_model, f'{LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}') 
        tokenizer = AutoTokenizer.from_pretrained(f'{LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}', padding_side='left')


    with open(f'{PATH_TO_DSET}/{args.dset}.json', 'r') as f:
        dset = json.loads(f.read())
    
    dir_to_save = f'next_token_pred/{args.corrupted_model_name}.npy'
    os.makedirs('next_token_pred', exist_ok=True)


    results, ppx, token, length = evaluate_model_next_token_pred(
        model=model,
        tokenizer=tokenizer,
        device=f'cuda:{args.cuda}',
        seed=args.seed,
        dataset=dset,
        dir_to_save=dir_to_save,
    )


    with open(f'next_token_pred/clean_{args.dset}.json', 'w') as f:
        f.write(json.dumps({
            'dset': args.dset,
            'ppx': str(float(ppx)),
        }))

    save(results, dir_to_save)
