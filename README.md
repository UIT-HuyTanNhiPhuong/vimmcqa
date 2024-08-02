# VIM-MCQA
Stand for Vietnamese-Medical-MultipleChoice-QuestionAnswering

## Description
## Installation
- Clone repo
```
!git clone https://github.com/UIT-HuyTanNhiPhuong/vimmcqa
```
- Download `requirements.txt`
```
%cd vimmcqa
!pip install -r requirements.txt
```
## Run Script

- `hf_token`: Your huggingface token (view on https://huggingface.co/settings/tokens)
- `prompt_mode`: `few_shot` or `zero_shot` (default value is `few_shot`)
- `model_name`: model's name on huggingface
- `test_first`: Set to True to run a test before the main execution (default is `True`)
- `test_index`: Limits the number of test iterations to this value when test_first is `True`.
```
!python main.py \
--hf_token <your_hf_token> \
--csv_file vimmcqa_test.csv \
--prompt_mode few_shot \
--model_name google/gemma-2-9b \
--test_first True \
--test_index 5
```


