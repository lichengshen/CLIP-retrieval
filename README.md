### Reproducing image-text retrieval results for CLIP (and variants)
- Tested on Karpathy split test sets following papers

#### Datasets
- Download COCO 2014 val images
- Get Karpathy split annotatons with `wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json`

#### Setup
`pip install -r requirements.txt`

#### Evaluate
`python eval.py`
|                       | COCO I2T R@1 | COCO I2T R@5 | COCO I2T R@10 | COCO T2I R@1 | COCO T2I R@5 | COCO T2I R@10 |
|-----------------------|--------------|--------------|---------------|--------------|--------------|---------------|
| CLIP paper reported   | 58.4         | 81.5         | 88.1          | 37.8         | 62.4         | 72.2          |
| Reproduced results    | 58.96        | 82.18        | 88.54         | 36.33        | 61.08        | 71.22         |
