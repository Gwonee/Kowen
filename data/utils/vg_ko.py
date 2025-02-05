import json
from tqdm import tqdm

with open("/data/ephemeral/home/qwen2_vl/LLaMA-Factory/data/coco_ko.json","r") as vq_ko:
    json_file = json.load(vq_ko)

num_image_2 = []
for i in tqdm(json_file):
    num_images = 0
    for j in i['conversations']:
        if '<image>' in j['value']:
            num_images += 1

    if num_images != 1:
        num_image_2.append(i['id'])
        print(i['id'], 'num images:',num_images)

filtered_data = []
for i in tqdm(json_file):
    if i['id'] not in num_image_2:
        filtered_data.append(i)

with open("/data/ephemeral/home/qwen2_vl/LLaMA-Factory/data/filtered_coco_ko.json", "w") as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=2)

    print("삭제 완료. 업데이트된 JSON 파일이 저장되었습니다.")
