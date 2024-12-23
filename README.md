### Reproducing image-text retrieval results for CLIP (and variants)
- Tested on Karpathy split test sets following papers

#### Datasets
- Download COCO 2014 val images
- Get Karpathy split annotatons with `wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json`

#### Setup
`pip install -r requirements.txt`

#### Evaluate
`python eval.py`