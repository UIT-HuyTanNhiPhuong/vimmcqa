from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset import Prompting_Dataset, handle_output
import torch
from tqdm import tqdm
import json
from torch.cuda.amp import autocast

import argparse
parser = argparse.ArgumentParser()

# Add argumentss
parser.add_argument('--hf_token', type=str)
parser.add_argument('--csv_file', type=str, default = 'finalData.csv')
parser.add_argument('--prompt_mode', type=str, default='zero_shot', 
                    help="'zero_shot': 0; 'one_shot': 1; 'two_shot': 2; 'three_shot': 3; 'few_shot': min(3, len(data) - 1)")
parser.add_argument('--model_name', type=str)
parser.add_argument('--max_length', type=int, default = 2000)
parser.add_argument('--test_first', type=bool, default = False)
parser.add_argument('--test_index', type=int, default = 5)
parser.add_argument('--output_file', type=str, default='results.json')

args = parser.parse_args()

import os
os.environ["HF_TOKEN"] = args.hf_token

# Dataset Loading
data = Prompting_Dataset(args.csv_file, args.prompt_mode)

# Evaluation Loop
exact_match = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_results = []  


# Load Model
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config)


# Evaluation Loop
for index, item in enumerate(tqdm(data)):
    # if args.test_first and index == args.test_index:
    #     break

    input_text, result = item['Prompt'], item['Result']
    input_ids = tokenizer(input_text, return_tensors="pt", truncation = True, max_length = args.max_length).to(device)

    with torch.no_grad(), autocast():
        outputs = model.generate(**input_ids, max_new_tokens=15)
        raw_output = tokenizer.decode(outputs[0][1:])
        
        processed_output = handle_output(raw_output, input_text)
        print('====================================')
        print(f'Ground_truth : {result}')
        print(f'Prediction : {processed_output}')
    
    res = {
        'index' : index,
        'output' : processed_output,
        'label' : result
    }
    save_results.append(res)

        
    # Free memory
    del input_ids
    del outputs
    del raw_output
    del processed_output
    del res
    torch.cuda.empty_cache()
    
    if index == len(data) -1: break # Ignore Key Error

with open(args.output_file, 'w') as json_file:
    json.dump(save_results, json_file, indent=4)

print(f"JSON files with prediction and ground truth data saved as {args.output_file}")