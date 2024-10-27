import openai
import os
import json
from gpt_cost_estimator import CostEstimator
from openai import OpenAI
from tqdm import tqdm
import argparse


COST_PER_TOKEN = 0.5e-6
JUDGE_PROMPT_VERSION = 1
JUDGE_MODEL = "gpt-3.5-turbo"

ROOT_PATH = 'evaluation/outputs'
PATH_TO_ANSWER_KEY = 'datasets/facts.json'

with open(f'chat_gpt_prompts/v{JUDGE_PROMPT_VERSION}.txt', 'r') as f:
    context = f.read()

client = OpenAI(api_key='')


def query_judge(prompt):
    response = client.chat.completions.create(
        model = JUDGE_MODEL,
        logprobs = False,
        messages=[{"role": "user", "content": prompt,}])

    output = response.choices[0].message.content
    tot_num_of_tokens = response.usage.total_tokens
    query_cost = tot_num_of_tokens * COST_PER_TOKEN
    
    return output, query_cost


def evaluate_judge_output(output: str):
    atoms = json.loads(output)
    correct, wrong = 0, 0

    result = {}
    logs = []

    for atom in atoms:
        logs.append(f'[JUDGE]: {atom["judgment"]} || [QUESTION]: {atom["question"]} | [OUTPUT]: {atom["output"]} || [ANSWERS]: {atom["answers"]}')
        
        if 'Accept' in atom["judgment"]:
            correct += 1
        elif 'Reject' in atom["judgment"]:
            wrong += 1
    
    assert correct + wrong == len(atoms)

    result = {
        'accuracy': f'{100.*correct/(correct + wrong):.2f}',
        'details': logs,
        'judge_outputs': output,
    }

    return result



all_outputs = {}
tot_cost = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code to evaluate models on dataset")
    parser.add_argument('--path_to_dset', type=str)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--run', type=int)
    
    args = parser.parse_args()

    which_run_to_consider = args.run

    with open(PATH_TO_ANSWER_KEY, 'r') as f:
        answer_key_dset = json.loads(f.read())

    with open(f'{ROOT_PATH}/{args.path_to_dset}', 'r') as f:
        evaluation_dset = json.loads(f.read())

    answer_key_dset = answer_key_dset['dset']

    for entity in tqdm(evaluation_dset):
        prompt_entity = '[\n'
        answer_key_dset_entity = answer_key_dset[entity]
        evaluation_dset_entity = evaluation_dset[entity]

        for pid in evaluation_dset_entity:
            atom = evaluation_dset_entity[pid]

            question = atom['question']
            output = atom['outputs'][which_run_to_consider]
            answers = answer_key_dset_entity[pid]['answers']
            
            prompt_entity += '{\n'
            prompt_entity += f'"question": "{question}",\n'
            prompt_entity += f'"output": "{output}",\n'
            prompt_entity += f'"answers": "{answers}"\n'
            prompt_entity += '},\n'
        
        prompt_entity += ']\n'

        prompt_to_judge = context + prompt_entity
        valid_gpt_output = False
        num_of_failed_attempts = 0

        while not valid_gpt_output:
            output, cost = query_judge(prompt=prompt_to_judge)
            tot_cost += cost

            try:
                all_outputs[entity] = evaluate_judge_output(output)
                valid_gpt_output = True
            except:
                num_of_failed_attempts += 1
                prompt_to_judge = context + prompt_entity + 'Make sure that you output in correct JSON format.\n'
                if num_of_failed_attempts >= 5:
                    break


    with open(f'judge_outputs/{args.filename}', 'w') as f:
        f.write(json.dumps({
            'cost': tot_cost,
            'result': all_outputs,
        }, indent=4))


    tot_questions = 0
    tot_accuracy = 0

    for entity in all_outputs:
        result = all_outputs[entity]
        curr_acc = float(result['accuracy'])
        num_of_questions = len(result['details'])

        tot_accuracy += curr_acc * num_of_questions
        tot_questions += num_of_questions
    
    tot_accuracy /= tot_questions

    with open(f'judge_outputs/{args.filename}', 'w') as f:
        f.write(json.dumps({
            'cost': tot_cost,
            'accuracy': f'{tot_accuracy:.3f}',
            'result': all_outputs,
        }, indent=4))

