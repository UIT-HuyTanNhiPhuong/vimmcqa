import json
import glob
import fnmatch
import argparse
from collections import Counter

# Parse command line arguments
import argparse

# Initialize ArgumentParser
parser = argparse.ArgumentParser(
    description="This script calculates performance metrics from JSON files containing prediction and ground truth data."
)

# Add arguments
parser.add_argument(
    '--json_directory', 
    type=str, 
    default='result.json',
    help=(
        "Path to the JSON file or directory containing JSON files with prediction and ground truth data. "
        "If a directory is specified, it should contain files matching the pattern 'result*.json'. "
        "If a single file is specified, it should be named 'result.json'. The default is 'result.json'."
    )
)

parser.add_argument(
    '--score_directory', 
    type=str, 
    default='final_score.json',
    help=(
        "Path to the output JSON file where the calculated scores (em, precision, recall, F1 score, MSE) will be saved. "
        "Default is 'final_score.json'."
    )
)

# Parse arguments
args = parser.parse_args()


def calculate_f1(gt, pred):
    tp = fp = fn = em = 0

    for g, p in zip(gt, pred):
        gt_set = set(g.split(", "))
        pred_set = set(p.split(", "))

        em += int(gt_set == pred_set)
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return em / len(gt), precision, recall, f1_score   

def convert_to_numeric(categories):
    category_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    return [category_to_num[item] for item in sum([list(set(cat)) for cat in categories], []) if item in category_to_num]

def calculate_mse(gt, pred):
    gt_numeric = convert_to_numeric(gt)
    pred_numeric = convert_to_numeric(pred)
    
    if not gt_numeric or not pred_numeric:
        return float('inf')  # To handle cases where lists might be empty

    squared_diffs = [(g - p) ** 2 for g, p in zip(gt_numeric, pred_numeric)]
    return sum(squared_diffs) / len(squared_diffs)

def calculate_scores(gt, pred):
    em, precision, recall, f1_score = calculate_f1(gt, pred)
    mse = calculate_mse(gt, pred)

    scores = {'em': em, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'mse': mse}
    print(scores)
    return scores

def load_json_files(directory_pattern):
    json_files = glob.glob(directory_pattern)
    all_data = []

    for json_file in json_files:
        print(json_file)
        with open(json_file, 'r') as file:
            all_data.extend(json.load(file))  # Use extend to flatten the list

    return all_data

def simplify_output(output):
    import re
    return [', '.join(re.findall(r'[A-D]', line)) for line in output]

def main():
    # Determine JSON files to read
    if fnmatch.fnmatch(args.json_directory, "*result*.json"):
        all_data = load_json_files(args.json_directory)
    else:
        all_data = load_json_files(args.json_directory + '/result*.json')

    # Extract 'output' and 'label'
    data = {'outputs': [], 'label': []}
    for item in all_data:
        data['outputs'].append(item['output'].replace('</s', '').strip())
        data['label'].append(item['label'].replace('</s', '').strip())

    # Calculate scores
    gt, pred = simplify_output(data['label']), simplify_output(data['outputs'])
    scores = calculate_scores(gt, pred)

    # Save scores to file
    with open(args.score_directory, 'w') as json_file:
        json.dump(scores, json_file)

if __name__ == "__main__":
    main()
