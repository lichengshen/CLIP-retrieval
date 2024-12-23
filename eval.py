import torch
from dataset import CocoCaptionsKarpathy
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336")
parser.add_argument("--prepend_prompt", action="store_true")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(args.model).to(device)
processor = AutoProcessor.from_pretrained(args.model)

dataset = CocoCaptionsKarpathy(
    root = "coco",
    annFile="coco/annotations/coco_karpathy_test.json"
)

batch_size = 32
image_features = []
text_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_images = []
        
        for j in range(i, min(i + batch_size, len(dataset))):
            image, _ = dataset[j]
            batch_images.append(image)
        
        image_inputs = processor(images=batch_images, return_tensors="pt").to(device)
        image_features.append(model.get_image_features(**image_inputs))
    
    for i in tqdm(range(0, len(dataset.text), batch_size)):
        batch_text = dataset.text[i:i + batch_size]
        for j in range(len(batch_text)):
            if args.prepend_prompt:
                batch_text[j] = "a photo of " + batch_text[j].lower()
            else:
                batch_text[j] = batch_text[j]
        text_inputs = processor(batch_text, return_tensors="pt", padding=True).to(device)
        text_features.append(model.get_text_features(**text_inputs))

    image_features = torch.cat(image_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    text_features = torch.cat(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = image_features @ text_features.T

    # I2T
    for k in [1, 5, 10]:
        correct = 0

        for i in range(len(dataset)):
            topk = similarities[i].topk(k).indices
            for txt_id in dataset.img2txt[i]:
                if txt_id in topk:
                    correct += 1
                    break
        
        print("I2T Recall@%d: %.2f" % (k, 100 * correct / len(dataset)))

    # T2I
    for k in [1, 5, 10]:
        correct = 0

        for i in range(len(dataset.text)):
            topk = similarities[:, i].topk(k).indices
            if dataset.txt2img[i] in topk:
                correct += 1
        
        print("T2I Recall@%d: %.2f" % (k, 100 * correct / len(dataset.text)))
        
