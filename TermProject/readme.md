# README

#### This GitHub page is a project that references [Meta-IFD](https://github.com/GISec-Team/Meta-IFD). It was primarily developed to fulfill a school assignment.

## 資料集(data和dataset的內容)請自行下載，附上原本的readme
**資料集下載完應如下 :** 
- data/Phish/processed/data.pt , pre_filter , pre_transform 
- data/Ponzi/processed/data.pt , pre_filter , pre_transform 
- dataset/Phish_unbalance, Ponzi_unbalance, node_hete_CA.csv, node_hete_EOA.csv 
- dataset/Phish_unbalance/edge/call_ca_ca.csv, call_eoa_ca.csv, trans_ca_ca.csv, trans_ca_eoa.csv , trans_eoa_ca.csv , trans_eoa_eoa.csv
- dataset/Phish_unbalance/CA.csv, EOA_labeled.csv, EOA_unlabeled.csv
- dataset/Ponzi_unbalance/edge/call_ca_ca.csv, call_eoa_ca.csv, trans_ca_ca.csv, trans_ca_eoa.csv , 
trans_eoa_ca.csv , trans_eoa_eoa.csv
- dataset/Ponzi_unbalance/CA.csv, EOA_labeled.csv, EOA_unlabeled.csv

## TermProject相關連結 (簡報、論文、Demo vedio)
雲端: https://drive.google.com/drive/folders/1ieKEYyvMFj9QBC8JwrpJRMUsj1KruDFY?usp=drive_link

## Meta-IFD

This repository provides a reference implementation of Metal-FD as described in the paper:

> **Enhancing Ethereum Fraud Detection via Generative and Contrastive Self-supervision**<br>

Available at [link](https://arxiv.org/abs/2408.00641).

## Dependencies

Recent versions of the following packages for Python are required:

* pytorch 2.1.1
* toch-geometric 2.4.0
* numpy 1.24.1
* pandas 2.0.3
* scikit-learn 1.3.2

## Datasets

Download data in PYG format from this [_notion page_](https://jjzhou.notion.site/Ethident-Data-f2a97c9d3ae74e11ae709e79fd56557d?pvs=4) and place it under the `/data` path.
You can also download the raw data, which is in the `/dataset` path.
`/dataset/Ponzi_unbalance` and `/dataset/Phish_unbalance` are raw data about building heterogeneous graphs.

The code for data processing is in `dataset.py`, including data cleaning and graph modeling,
and saves the graph to `/data`.


## Usage

1. Run `icvae_pertrain.py` to pretrain the generative model, and save the model to `/pretrain_model/`.
2. Run `run.py` to execute the multi-view Ethereum interaction feature learning and fraud detection, 

```bash
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 1 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 1 

python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 8 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 4 --lr 0.02 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 10 
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 128 --epochs 10
python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 2 --batch_size 256 --epochs 10

python run.py  --dataset Phish --hidden 4 --lr 0.01 --loss_train 0.5 --concat 1 --batch_size 128 --epochs 5
```

## Reference
````bash
@article{jin2024enhancing,
  author={Jin, Chengxiang and Zhou, Jiajun and Xie, Chenxuan and Yu, Shanqing and Xuan, Qi and Yang, Xiaoniu},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Enhancing Ethereum Fraud Detection via Generative and Contrastive Self-Supervision}, 
  volume={20},
  number={},
  pages={839-853},
  year={2024},
  doi={10.1109/TIFS.2024.3521611}
}
````
