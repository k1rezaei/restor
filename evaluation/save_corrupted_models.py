from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
import torch


CLEAN_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
LORA_PATH = ''
MODEL_PATH = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    
    parser.add_argument('--corrupted_model_path', type=str,)
    parser.add_argument('--corrupted_model_name', type=str,)

    args = parser.parse_args()

    base_model = LlamaForCausalLM.from_pretrained(CLEAN_MODEL_NAME)
    lora_path = f'{LORA_PATH}/{args.corrupted_model_path}'
    model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    model = model.to('cpu')
    model = model.merge_and_unload()

    path_to_save_model = f'{MODEL_PATH}/{args.corrupted_model_name}/'
    
    model.save_pretrained(path_to_save_model)
    tokenizer.save_pretrained(path_to_save_model)

    print(f"Merged model saved to {path_to_save_model}")
