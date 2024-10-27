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
gc.enable()

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


def evaluate_model_opinion_on_fact_dataset(model, tokenizer, device, all_data, options, seed, entities_list=[], batch_size=1):
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
    
    for entity in tqdm(entities_list):
        results[entity] = {}
        for pid in dset[entity]:
            question = dset[entity][pid]['question']
            
            if pid not in options[entity]:
                continue
        
            rel_context = get_context(pid=pid, context=context, qid=entity)
            prompt = rel_context + f'{question}'
            
            results[entity][pid] = {}

            for category in ['correct', 'corrupted', 'random']:
                all_answers = options[entity][pid][category]
                normalized_probs = []
                
                for t in range(0, len(all_answers), batch_size):
                    answers = all_answers[t: min(len(all_answers), t + batch_size)]

                    qa = [prompt + ' ' + answer for answer in answers]
                    qa_tokenized = tokenizer(qa, return_tensors="pt", padding=True)
                    labels = qa_tokenized.input_ids.to(device)

                    answer_ids = [tokenizer(' ' + answer, return_tensors="pt").input_ids[0] for answer in answers]

                    for i in range(len(answers)):
                        st = -(len(answer_ids[i]) - 1)
                        labels[i, 0: st] = -100

                        assert (answer_ids[i][1:] - qa_tokenized.input_ids[i][st:]).sum() < 0.01

                    
                    qa_tokenized = {key: value.to(device) for key, value in qa_tokenized.items()}
                    outputs = model(**qa_tokenized, labels=labels)

                    logits = outputs.logits
                    loss = outputs.loss.item()
                    sanity = 0
                    
                    for i in range(len(answers)):
                        st = -(len(answer_ids[i]) - 1)

                        curr_logits = logits[i, st-1: -1]
                        log_probs = torch.log_softmax(curr_logits, dim=-1)

                        answer_token_ids = qa_tokenized['input_ids'][i, st:]
                        answer_log_probs = log_probs[range(len(answer_ids[i]) - 1), answer_token_ids]

                        sanity += answer_log_probs.sum().item()
                        normalized_probs.append(answer_log_probs.mean().item())


                    num_of_tokens = (labels != -100).sum().item()

                    assert (sanity / num_of_tokens + loss) ** 2 < 0.01

                results[entity][pid][category] = {
                    'options': all_answers,
                    'probabilty (normalized)': normalized_probs,
                    'average': sum(normalized_probs) / len(normalized_probs)
                }

    return results
