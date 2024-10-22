# Grounding DINO Zira
---
Official pytorch implementation of [ZiRa](https://arxiv.org/abs/2403.01680), a method for incremental vision language object detection (IVLOD)

## Install 
If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set.

```bash
pip install -e .
```

## Dataset
Download the coco dataset in the "dataset" folder, and download the all Odinw13 sub-datasets in a folder named "dataset". The Odinw13 sub-datasets can be downloaded by runing this script.

```bash
python download.py
```

## Training
Run the scripts.

```bash
sh train_odinw13_zira.sh
```

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{DBLP:journals/corr/abs-2403-01680,
  author       = {Jieren Deng and
                  Haojian Zhang and
                  Kun Ding and
                  Jianhua Hu and
                  Xingxuan Zhang and
                  Yunkuan Wang},
  title        = {Zero-shot Generalizable Incremental Learning for Vision-Language Object
                  Detection},
  journal      = {CoRR},
  volume       = {abs/2403.01680},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2403.01680},
  doi          = {10.48550/ARXIV.2403.01680},
  eprinttype    = {arXiv},
  eprint       = {2403.01680},
  timestamp    = {Tue, 02 Apr 2024 16:35:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2403-01680.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```




