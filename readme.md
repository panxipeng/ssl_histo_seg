# UTCS: Uncertainty-guided Cross Teaching Semi-supervised Framework forHistopathology Tissue Segmentation with Curriculum Self-training (Under Review)

## Environment
To run this project, you need:
* Ubuntu 20.04 system
* Python 3.8
* PyTorch (Version 1.11 or higher)
* At least 12GB GPU memory
* Install all dependencies: `pip install -r requirements.txt`

## Data
* BCSS and LUAD-HistoSeg datasets can found in [this link](https://github.com/ChuHan89/WSSS-Tissue).
* Data format:
   ```
  ssl_histo_seg/data/
      JPEGImages/
        TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png
        ...
      SegmentationClass/
        TCGA-GM-A2DD-DX1_xmin47260_ymin22408_MPP-0.2500+247.png
        ...
   ```  
* Different proportions of labeled/unlabeled data and validation data are avaliable in `ssl_histo_seg/bcss_split` and `ssl_histo_seg/luad_split`. Before training, you should put them into `dataset/splits/pascal`.

## Pretrained Weights
Download from [Baidu Cloud](https://pan.baidu.com/s/1t-yQBDSsciHdKHqnjeyCRQ?pwd=3pvb) or [Google Drive](https://drive.google.com/drive/folders/1-PTL1p30yz-a-NUYN8nWWkruFraoHk1w?usp=sharing), and put them into ``ssl_histo_seg/pretrained``.

## Training
```
python main.py --plus True --reliable-id-path outdir/reliable_id_path  --dataset pascal --data-root data --batch-size 16 --backbone resnet101 --model deeplabv3plus --labeled-id-path dataset/splits/pascal/1_8/split_0/labeled.txt --unlabeled-id-path dataset/splits/pascal/1_8/split_0/unlabeled.txt --pseudo-mask-path outdir/pseudo_masks/pascal/1_8/split_0  --save-path outdir/models/pascal/1_8/split_0
```

