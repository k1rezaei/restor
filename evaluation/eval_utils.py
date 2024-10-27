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

def get_context(pid, context, qid, num_of_examples=5):
    output = ''
    info = context[pid]

    for entry in info:
        if entry['entity'] == qid:
            continue
        
        output += f'{entry["question"]} {entry["answer"]}\n'
        num_of_examples -= 1
        if num_of_examples == 0:
            return output

    return output


def evaluate_model_on_fact_dataset(model, tokenizer, device, all_data, seed, batch_size, max_length, number_of_repeats=1, entities_list=[]):
    dset = all_data['dset']
    context = all_data['context']

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()

    results = {}

    if len(entities_list) == 0:
        entities_list = list(dset.keys())
    

    whose_pid, whose_entity = {}, {}

    print('step 1 (doing)\t[aggregating prompts over all (entity, properties)]')

    prompts = []
    counter = 0

    for entity in entities_list:
        results[entity] = {}
        for pid in dset[entity]:
            question = dset[entity][pid]['question']
            rel_context = get_context(pid=pid, context=context, qid=entity)
            prompt = rel_context + f'{question}'
            for _ in range(number_of_repeats):
                prompts.append(prompt)
                whose_entity[counter] = entity
                whose_pid[counter] = pid
                counter += 1        

    print(f'step 1 (done)\t[total # of prompts: {counter}]')
    

    num_of_batches = (counter + batch_size - 1)// batch_size
    print(f'step 2 (doing)\t[total # of batches: {num_of_batches}]')

    with torch.no_grad():
        
        for batch_id in tqdm(range(num_of_batches)):
            i = batch_id * batch_size

            batch_prompts = prompts[i: min(i+batch_size, len(prompts))]
                
            if batch_size == 1:
                inputs = tokenizer(batch_prompts, return_tensors="pt").to(device)
            else:
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                        
            generate_ids = model.generate(
                **inputs,
                min_new_tokens = 2, #1,
                pad_token_id=tokenizer.eos_token_id,
                max_length=max_length,
                do_sample=True)
            
            outputs = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)


            for j, prompt in enumerate(batch_prompts):
                pid = whose_pid[i + j]
                entity = whose_entity[i + j]

                if pid not in results[entity]:
                    results[entity][pid] = {
                        'question': dset[entity][pid]['question'],
                        'prompt': batch_prompts[j],
                        'outputs': [],
                        'raw_outputs': [],
                    }

                
                curr_output = outputs[j]
                curr_output = curr_output[len(prompt) + 1: ]
                results[entity][pid]['raw_outputs'].append(curr_output)
                
                if curr_output.find('\n') != -1:
                    curr_output = curr_output[:curr_output.find('\n')]
                
                if len(curr_output) == 0:
                    print('[WARNING] -- there is empty string in the output.')
                
                results[entity][pid]['outputs'].append(curr_output)

    print(f'step 2 (done)\t[calculated all outputs with {number_of_repeats} replications]')
    return results
    
