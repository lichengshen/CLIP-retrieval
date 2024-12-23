# Reference: https://github.com/salesforce/BLIP/blob/main/data/coco_karpathy_dataset.py
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class CocoCaptionsKarpathy(Dataset):
    def __init__(self, root, annFile):
        self.annotation = json.load(open(annFile, "r"))
        self.root = root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                self.text.append(caption)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        target = self.annotation[index]["caption"]

        return image, target