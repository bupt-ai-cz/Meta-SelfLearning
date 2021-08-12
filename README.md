## Meta-SelfLearning
Meta Self-learning for Multi-Source Domain Adaptation： A Benchmark

[Project](https://bupt-ai-cz.github.io/Meta-SelfLearning/)

## Data Prepare
Before using the raw data, you need to convert it to lmdb dataset.
```
python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
The data folder should be organized below
```
data
├── train_label.txt
└── imgs
    ├── 000000001.png
    ├── 000000002.png
    ├── 000000003.png
    └── ...
```
The format of train_label.txt should be `{imagepath}\t{label}\n`
For example,
```
imgs/000000001.png Tiredness
imgs/000000002.png kills
imgs/000000003.png A
```

## Requirements
* Python == 3.7
* Pytorch == 1.7.0
* torchvision == 0.8.1

## Argument
* `--train_data`: folder path to training lmdb dataset.
* `--valid_data`: folder path to validation lmdb dataset.
* `--select_data`: select training data, examples are shown below
* `--batch_ratio`: assign ratio for each selected data in the batch. 
* `--Transformation`: select Transformation module [None | TPS], in our method, we use None only.
* `--FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet], in our method, we use ResNet only.
* `--SequenceModeling`: select SequenceModeling module [None | BiLSTM], in our method, we use BiLSTM only.
* `--Prediction`: select Prediction module [CTC | Attn], in our method, we use Attn only.
* `--saved_model`: path to a pretrained model.
* 


## Get started
#### To train the baseline model for synthetic domain.
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train.py \
    --train_data data/train/ \
    --select_data car-doc-street-handwritten \
    --batch_ratio 0.25-0.25-0.25-0.25 \
    --valid_data data/test/syn \
    --Transformation None --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction Attn \
    --batch_size 96 --valInterval 5000
```

#### To train the meta_train model for synthetic domain using the pretrained model.
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python meta_train.py 
    --train_data data/train/ \ 
    --select_data car-doc-street-handwritten \
    --batch_ratio 0.25-0.25-0.25-0.25 \
    --valid_data data/test/syn/ \
    --Transformation None --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction Attn \
    --batch_size 96  --source_num 4  \
    --valInterval 5000 --inner_loop 1 --valInterval 5000 \
    --saved_model saved_models/pretrained.pth 
```

#### To train the pseudo-label model for synthetic domain.
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python self_training.py 
    --train_data data/train \
    —-select_data car-doc-street-handwritten \
    --batch_ratio 0.25-0.25-0.25-0.25 \
    --valid_data data/train/syn \
    --test_data data/test/syn \
    --Transformation None --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction Attn \
    --batch_size 96  --source_num 4 \
    --warmup_threshold 28 --pseudo_threshold 0.9 \
    --pseudo_dataset_num 50000 --valInterval 5000 \ 
    --saved_model saved_models/pretrained.pth 
```
#### To train the meta self-learning model for synthetic domain.
```
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python meta_self_learning.py 
    --train_data data/train \
    —-select_data car-doc-street-handwritten \
    --batch_ratio 0.25-0.25-0.25-0.25 \
    --valid_data data/train/syn \
    --test_data data/test/syn \
    --Transformation None --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction Attn \
    --batch_size 96 --source_num 4 \
    --warmup_threshold 0 --pseudo_threshold 0.9 \
    --pseudo_dataset_num 50000 --valInterval 5000 --inner_loop 1 \
    --saved_model pretrained_model/pretrained.pth 
```
## Contact
* email: qiushuhao@bupt.edu.cn; czhu@bupt.edu.cn
