# Evaluation Framework for CCMusic Database MSA
[![Python application](https://github.com/monet-joe/ccmusic_eval/actions/workflows/python-app.yml/badge.svg?branch=msa)](https://github.com/monet-joe/ccmusic_eval/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/monet-joe/ccmusic_eval/blob/msa/LICENSE)

## Download
```bash
git clone -b msa git@github.com:monet-joe/ccmusic_eval.git
cd ccmusic_eval
```

## Requirements
```bash
conda create -n cv --yes --file conda.txt
conda activate cv
pip install -r requirements.txt
```

## Usage
1. run beat_track.py first to get beat information, saved to 'dataset\references'
2. run process.py to perform structure analysis using beat information from 'dataset\references'
3. run txt_to_lab.py to transform .txt to .lab as mir_eval need .lab
4. run eval.py to evaluate and plot results

## Cite
```
@dataset{zhaorui_liu_2021_5676893,
  author       = {Monan Zhou, Shenyang Xu, Zhaorui Liu, Zhaowen Wang, Feng Yu, Wei Li and Zijin Li},
  title        = {CCMusic: an Open and Diverse Database for Chinese and General Music Information Retrieval Research},
  month        = {nov},
  year         = {2021},
  publisher    = {Zenodo},
  version      = {1.1},
  doi          = {10.5281/zenodo.5676893},
  url          = {https://doi.org/10.5281/zenodo.5676893}
}
```