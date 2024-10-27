from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
from tqdm import tqdm
import os
import json
import argparse

import torch
import gc
torch.cuda.empty_cache()

from mcqa_utils import evaluate_model_opinion_on_fact_dataset


CLEAN_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
OUTPUT_NAME = 'Llama3-clean'
PATH_TO_EVAL = 'evaluation/datasets'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dset', type=str, default='all_entities_eval_dset.json')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--path_to_eval', type=str, default=PATH_TO_EVAL)
    parser.add_argument('--path_to_entities', type=str, default='')
    parser.add_argument('--path_to_categories', type=str, default='')
    parser.add_argument('--filename', type=str,)
    parser.add_argument('--st', type=int,)
    parser.add_argument('--en', type=int,)
    
    args = parser.parse_args()

    model = LlamaForCausalLM.from_pretrained(CLEAN_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(CLEAN_MODEL_NAME, padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    
    with open(f'{args.path_to_eval}/{args.dset}', 'r') as f:
        all_data = json.loads(f.read())

    with open(f'{args.path_to_eval}/{args.path_to_categories}', 'r') as f:
        options = json.loads(f.read())

    try:
        with open(f'{args.path_to_eval}/{args.path_to_entities}', 'r') as f:
            entities_list = json.loads(f.read())
    except:
        entities_list = []

    entities_list = entities_list[args.st: args.en]

    
    results = evaluate_model_opinion_on_fact_dataset(
        model=model,
        tokenizer=tokenizer,
        device=f'cuda:{args.cuda}',
        all_data=all_data,
        options=options,
        seed=args.seed,
        entities_list=entities_list,
    )


    dir_to_save = f'mcqa_outputs/{OUTPUT_NAME}'
    filename = args.filename
    if len(filename) == 0:
        filename = args.dset

    os.makedirs(dir_to_save, exist_ok=True)
    with open(f'{dir_to_save}/st_{args.st}_en_{args.en}_{filename}', 'w') as f:
        f.write(json.dumps(results, indent=4))
