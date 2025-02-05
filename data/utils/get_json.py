import json

from datasets import load_dataset
from transformers import AutoProcessor
from data_utils import  K_DTCBench_for_train

class prepare_data():
    def __init__(self,dataset):
        self.dataset_name = dataset

        if dataset == "K-DTCBench":
            self.ds = load_dataset("NCSOFT/K-DTCBench")
            

    def save_image(self):
        print('start save image')
        for i in self.ds['test']:
            id = i['index']
            i['image'].save('./LLaMA-Factory/data/' + self.dataset_name + '/' + id + '.jpg')

    def get_json(self):
        messages = []
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        train_dataset = K_DTCBench_for_train(dataset = self.ds, processor = processor)
        
        print('start generate prompt')
        for i in self.ds['test']:
            message, label = train_dataset.generate_prompt(i)
            messages.append(message)
        
        print('start save json')
        with open("./LLaMA-Factory/data/"+self.dataset_name+".json", "w", encoding="utf-8") as json_file:
            json.dump(messages, json_file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    data = prepare_data(dataset='K-DTCBench')
    data.save_image()
    data.get_json()




