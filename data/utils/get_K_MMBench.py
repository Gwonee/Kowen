import os
import torch
from torch.utils.data import Dataset,DataLoader

from qwen_vl_utils import process_vision_info

class K_MMBench(Dataset):
    def __init__(self,dataset,processor):
        self.ds = dataset['dev']
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self,idx):
        data = self.ds[idx]

        index = data['index']
        answer = data['answer']
        messages, answer = self.generate_prompt(data)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        return inputs, index, answer
    
    def generate_prompt(self, data):
        questions = ['question','hint']
        options = ['A','B','C','D']
        text = []
        for i in questions:
            input = data[i]
            if input is not None:
                text.append(i+': '+input)
        text.append('Options:')
        for i in options:
            input = data[i]
            if input is not None:
                text.append(i+'. '+input)
        text.append('주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.')
        text_prompt = '\n'.join(text)
        image = data['image']
        answer = data['answer']

        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": text_prompt},
                ]
            }
        ]
        return prompt, answer

    def process_function(self, examples):
        messages, answer = self.generate_prompt(examples)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        labels = self.processor(text=answer,padding=True,truncation=True,return_tensors="pt")

        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        inputs['pixel_values'] = inputs['pixel_values'][0]
        inputs['image_grid_thw'] = inputs['image_grid_thw'][0]
        inputs['labels'] = labels['input_ids'][0]
        return inputs

class K_DTCBench(Dataset):
    def __init__(self,dataset,processor):
        self.ds = dataset['test']
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self,idx):
        data = self.ds[idx]

        index = data['index']
        answer = data['answer']
        messages, answer = self.generate_prompt(data)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        return inputs, index, answer
    
    def generate_prompt(self, data):
        questions = ['question']
        options = ['choice_a','choice_b','choice_c','choice_d']
        text = []
        Options = []
        for i in questions:
            input = data[i]
            if input is not None:
                text.append(input)
        Options.append('Options:')
        for i in options:
            input = data[i]
            opt = i[-1].upper()
            if input is not None:
                Options.append(opt+': '+input+',')
        option_prompt = ' '.join(Options)
        text.append(option_prompt)
        text.append('주어진 선택지 중 해당 옵션의 문자로 직접 답하세요.')
        text_prompt = '\n'.join(text)
        image = data['image']
        answer = data['answer']

        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": text_prompt},
                ]
            },
        ]
        return prompt, answer

    def process_function(self, examples):
        messages, answer = self.generate_prompt(examples)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        labels = self.processor(text=answer, padding=True, truncation=True)
        
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        inputs['pixel_values'] = inputs['pixel_values'][0]
        inputs['image_grid_thw'] = inputs['image_grid_thw'][0]
        inputs['labels'] = labels['input_ids'][0]
        return inputs