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

UNLEARNED_MODELS_PATH = ''
LIMA_LORA_PATH = ''
PATH_TO_EVAL = 'evaluation/datasets'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dset', type=str, default='all_entities_eval_dset.json')
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--path_to_eval', type=str, default=PATH_TO_EVAL)
    parser.add_argument('--path_to_categories', type=str, default='')
    parser.add_argument('--path_to_entities', type=str, default='')
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--corrupted_model_config', type=str,)
    parser.add_argument('--unlearning_configs', type=str,)
    parser.add_argument('--corrupted_dataset', type=str,)
    parser.add_argument('--lima_config', type=str, default='')
    parser.add_argument('--st', type=int,)
    parser.add_argument('--en', type=int,)
 
    args = parser.parse_args()

    print(f'{UNLEARNED_MODELS_PATH}/{args.corrupted_dataset}/{args.unlearning_configs}/{args.corrupted_model_config}')

    if len(args.lima_config) == 0:
        print("Original Unlearned Model")
        
        model = LlamaForCausalLM.from_pretrained(f'{UNLEARNED_MODELS_PATH}/{args.corrupted_dataset}/{args.unlearning_configs}/{args.corrupted_model_config}/checkpoints')
        tokenizer = AutoTokenizer.from_pretrained(f'{UNLEARNED_MODELS_PATH}/{args.corrupted_dataset}/{args.unlearning_configs}/{args.corrupted_model_config}/checkpoints', padding_side='left')
    
    else:
        print("LIMA FT Unlearned Model")
        print(f'LORA PATH: {LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}')

        base_model = LlamaForCausalLM.from_pretrained(f'{UNLEARNED_MODELS_PATH}/{args.corrupted_dataset}/{args.unlearning_configs}/{args.corrupted_model_config}/checkpoints')
        model = PeftModel.from_pretrained(base_model, f'{LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}') 
        tokenizer = AutoTokenizer.from_pretrained(f'{LIMA_LORA_PATH}/{args.corrupted_dataset}/{args.corrupted_model_config}_{args.lima_config}', padding_side='left')


    with open(f'{args.path_to_eval}/{args.dset}', 'r') as f:
        all_data = json.loads(f.read())
    
    im = 'mix_a4' if 'mix_a4' in args.corrupted_dataset else ('mix_a4_v4' if 'mix_a4_v4' in args.corrupted_dataset else ('corrupted_v2' if 'corrupted' in args.corrupted_dataset else 'squad_v7'))
    
    with open(f'{args.path_to_eval}/{im}/{args.path_to_categories}', 'r') as f:
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
        seed=args.seed,
        options=options,
        entities_list=entities_list,
        batch_size=4,
    )
    
    if len(args.lima_config) == 0:
        dir_to_save = f'mcqa_outputs/unlearned/{args.corrupted_dataset}/Llama3-{args.corrupted_model_config}'
    else:
        dir_to_save = f'mcqa_outputs/unlearned/{args.corrupted_dataset}/Llama3-{args.corrupted_model_config}_{args.lima_config}'
    
    filename = args.filename
    if len(filename) == 0:
        filename = args.dset


    os.makedirs(dir_to_save, exist_ok=True)
    with open(f'{dir_to_save}/st_{args.st}_en_{args.en}_{filename}.json', 'w') as f:
        f.write(json.dumps(results, indent=4))
    
