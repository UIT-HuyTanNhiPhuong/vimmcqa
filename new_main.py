from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset import Prompting_Dataset, handle_output
import torch
from tqdm import tqdm
import json
from torch.cuda.amp import autocast
import argparse
import os

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--hf_token', type=str)
parser.add_argument('--csv_file', type=str, default='finalData.csv')
parser.add_argument('--prompt_mode', type=str, default='zero_shot')
parser.add_argument('--model_name', type=str)
parser.add_argument('--test_first', type=bool, default=False)
parser.add_argument('--test_index', type=int, default=5)
parser.add_argument('--output_file', type=str, default='results.json')
parser.add_argument('--start_index', type=int, default=0)

args = parser.parse_args()

os.environ["HF_TOKEN"] = args.hf_token

# Dataset Loading
data = Prompting_Dataset(args.csv_file, args.prompt_mode)

# Evaluation Loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_results = []

# Load Model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config
)

# Evaluation Loop
for index in range(args.start_index, len(data)):
    input_text, result = data[index]['Prompt'], data[index]['Result']
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad(), autocast():
        outputs = model.generate(**input_ids, max_new_tokens=15)
        raw_output = tokenizer.decode(outputs[0][1:])

    processed_output = handle_output(raw_output, input_text)
    print('====================================')
    print(f'Ground_truth : {result}')
    print(f'Prediction : {processed_output}')
    res = {
        'index': index,
        'output': processed_output,
        'label': result
    }
    save_results.append(res)

    # Save results every 100 items
    if (index + 1) % 100 == 0:
        output_part_file = f"{args.output_file.split('.')[0]}_part_{index // 100}.json"
        with open(output_part_file, 'w') as json_file:
            json.dump(save_results, json_file, indent=4)
        save_results = []  # Reset results after saving

    # Free memory
    del input_ids
    del outputs
    del raw_output
    del processed_output
    del res
    torch.cuda.empty_cache()

    if index == len(data) -1: break # Ignore Key Error

# Save any remaining results
if save_results:
    with open(args.output_file, 'w') as json_file:
        json.dump(save_results, json_file, indent=4)

print(f"Results saved to {args.output_file} and parts every 100 items.")
