
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd

def handle_output(raw_output, prompt):
  start_index = raw_output.find(prompt)

  generated_part = raw_output[start_index + len(prompt):]

  start_index = generated_part.find('[')
  end_index = generated_part.find('\n\n')
  prediction = generated_part[start_index:end_index].strip()

  return prediction


class Prompting_Dataset(Dataset):
    def __init__(self, csv_file, prompt_mode='few_shot'):
        self.csv_file = csv_file
        self.prompt_mode = prompt_mode
        self.load_csv_file()

    def load_csv_file(self):
        self.data = pd.read_csv(self.csv_file)

    def add_prompt(self, sample, include_answer=False):
        prompt = 'Hãy chọn options đúng\n'
        context = sample['context']
        question = sample['question']
        prompt += f'Context : {context}\nQuestion : {question}\nChoices:\n'
        choices = ['A', 'B', 'C', 'D']
        text = [sample[each] for each in choices]

        for choice, txt in zip(choices, text):
            prompt += f"{choice}. {txt}\n"

        if include_answer:
            answer = "".join(sample['result'])
            prompt += f'Answer: {answer}\n\n'

        return prompt

    def create_prompt(self, sample):
        role_description = (
            "Role: Bạn là một chuyên gia y tế hàng đầu.\n"
            "Bạn có khả năng phân tích và trả lời các câu hỏi trắc nghiệm y sinh một cách chính xác, rõ ràng và dễ hiểu.\n"
            "Bạn sẽ xem xét kỹ lưỡng câu hỏi và tất cả các lựa chọn trả lời, sau đó đưa ra câu trả lời đúng nhất.\n"
            "Nếu có nhiều hơn một đáp án đúng, bạn sẽ liệt kê tất cả. Nếu bạn trả lời chính xác và đầy đủ, bạn sẽ được thưởng 200$.\n\n"
        )

        num_shots = {
            'zero_shot': 0,
            'one_shot': 1,
            'two_shot': 2,
            'three_shot': 3,
            'few_shot': min(3, len(self.data) - 1)  # Fallback for few_shot
        }

        # Get the number of examples to include
        shot_count = num_shots.get(self.prompt_mode, 0)

        # Build the prompt
        res = role_description

        for i in range(shot_count):
            sample_example = self.data.iloc[i]
            res += self.add_prompt(sample=sample_example, include_answer=True)

        # Add the main sample without an answer
        res += self.add_prompt(sample=sample, include_answer=False)
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.loc[idx]
        prompt = self.create_prompt(sample)
        result = self.data.loc[idx]['result']

        return {
            'Prompt': prompt,
            'Result': result,
        }

# class Prompting_Dataset(Dataset):
#     def __init__(self, csv_file, prompt_mode = 'few_shot'):
#       self.csv_file = csv_file
#       self.prompt_mode = prompt_mode
#       self.load_csv_file()

#     def load_csv_file(self):
#       import pandas as pd
#       self.data = pd.read_csv(self.csv_file)

#     def add_prompt(self, sample, few_shot = False):
#       prompt = ''
#       if few_shot:
#         context = sample['context']
#         question = sample['question']
#         prompt += f'Context : {context}\nQuestion : {question}\nChoices:\n'
#         choices = ['A', 'B', 'C', 'D']
#         text = [sample[each] for each in choices]

#         for choice, txt in zip(choices, text):
#           prompt += f"{choice}. {txt}\n"

#         answer = "".join(sample['result'])
#         prompt += f'Answer: {answer}\n\n'

#       else:
#         prompt = 'Hãy chọn options đúng\n'
#         context = sample['context']
#         question = sample['question']
#         prompt += f'Context : {context}\nQuestion : {question}\nChoices:\n'
#         choices = ['A', 'B', 'C', 'D']
#         text = [sample[each] for each in choices]

#         for choice, txt in zip(choices, text):
#           prompt += f"{choice}. {txt}\n"

#       return prompt
        

#     def create_prompt(self, sample):
#       res = f"Role: Bạn là một chuyên gia y tế hàng đầu.\nBạn có khả năng phân tích và trả lời các câu hỏi trắc nghiệm y sinh một cách chính xác, rõ ràng và dễ hiểu.\nBạn sẽ xem xét kỹ lưỡng câu hỏi và tất cả các lựa chọn trả lời, sau đó đưa ra câu trả lời đúng nhất.\nNếu có nhiều hơn một đáp án đúng, bạn sẽ liệt kê tất cả.Nếu bạn trả lời chính xác và đầy đủ, bạn sẽ được thưởng 200$.\n\n"
      
#       # res = f"""Role: Bạn là một chuyên gia y tế hàng đầu.\nBạn có khả năng phân tích và trả lời các câu hỏi trắc nghiệm y sinh một cách chính xác, rõ ràng và dễ hiểu.\nNhiệm vụ của bạn là sử dụng kiến thức y khoa sâu rộng để đọc hiểu, phân tích và trả lời các câu hỏi y khoa một cách chính xác. \nVui lòng xem xét kỹ lưỡng từng câu hỏi cùng với các lựa chọn A, B, C, D và sử dụng nguồn kiến thức y khoa phong phú của bạn để đưa ra câu trả lời đúng cho mỗi câu hỏi. \nMỗi câu hỏi có thể có từ 1 đến 4 đáp án đúng; không có câu nào không có đáp án đúng. \nHãy trả lời chính xác và đầy đủ, định dạng đầu ra của bạn phải là danh sách gồm các chữ cái đại diện cho đáp án đúng, ví dụ: "['A', 'B', 'C', 'D']", "['A', 'B', 'C']", "['A', 'B']", "['A', 'D']", "['A']". \nNếu bạn trả lời chính xác và đầy đủ, bạn sẽ được thưởng 200$.\n\n"""

#       if self.prompt_mode == 'few_shot':
#         sample_single = self.data.iloc[0]
#         sample_multiple = self.data.iloc[1]  

#         res += self.add_prompt(sample = sample_single, few_shot = True)
#         res += self.add_prompt(sample = sample_multiple, few_shot = True)

#       res += self.add_prompt(sample = sample, few_shot = False)
#       return res

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#       sample = self.data.loc[idx]

#       prompt = self.create_prompt(sample)
#       result = self.data.loc[idx]['result']

#       return {
#           'Prompt' : prompt,
#           'Result' : result,
#       }
