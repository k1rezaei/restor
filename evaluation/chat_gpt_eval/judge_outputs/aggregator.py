import os
import json
import argparse
import numpy as np

PATH_TO_JUDGE = 'evaluation/chat_gpt_eval/judge_outputs'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    parser.add_argument('--path_to_judge', type=str)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    
    new_results = {}
    num_of_questions = {}

    for i in range(args.runs):
        with open(f'{PATH_TO_JUDGE}/{args.path_to_judge}_run{i}.json', 'r') as f:
            outputs = json.loads(f.read())
            outputs = outputs['result']
        
        for entity in outputs:
            num_of_questions[entity] = len(outputs[entity]['details'])

            ls = new_results.get(entity)
            if ls is None:
                ls = []

            ls.append(float(outputs[entity]['accuracy']))
            new_results[entity] = ls

        
    
    for entity in new_results:
        new_results[entity] = {
            'accuracy': float(np.mean(np.array(new_results[entity]))),
            'num_of_questions': num_of_questions[entity],
            'list': new_results[entity],
        }
    
    with open(f'{PATH_TO_JUDGE}/{args.path_to_judge}.json', 'w') as f:
        f.write(json.dumps(new_results, indent=4))


