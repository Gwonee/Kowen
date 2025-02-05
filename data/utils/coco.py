import json
from PIL import Image
from transformers import AutoProcessor
import torch
from torch.nn.utils.rnn import pad_sequence

def load_dataset(json_path, coco_image_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def preprocess_function(example):
        image_path = f"{coco_image_path}/{example['image']}"
        image = Image.open(image_path).convert("RGB")

        text = ""
        for convo in example['conversations']:
            if convo['from'] == "human":
                text += convo['value'] + "\n"
            elif convo['from'] == "gpt":
                text += convo['value'] + "\n"

        inputs = processor(images=image, text=text, return_tensors="pt", padding=True)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
        }

    return [preprocess_function(example) for example in dataset[:1000]]

def collate_fn(batch):
    pixel_values = [b["pixel_values"] for b in batch]
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    pixel_values_padded = torch.stack(pixel_values)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "pixel_values": pixel_values,
    }