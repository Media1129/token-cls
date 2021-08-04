# Recipe entity recognition task

## Generate recipe entity data
1. data folder -> data_tag/
    + train_aug.json
    + dev.json
    + test.json
    + train.txt
    + dev.txt
    + test.txt
2. data_tag/generate_data.py
    + transform data_tag/*.json to data_tag/*.txt
3. recipe_dataset.py
    + setup features classlabel: O, dishname, ingredient
4. run.sh -> change the 314 line -> config:num_labels
```bash=
python3 run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name recipe_dataset.py \
  --output_dir ./result_v2/ \
  --do_train \
  --do_predict \
  --overwrite_output_dir \
  --num_train_epochs 20
```

5. predict.sh 
```bash=
python3 run_ner.py \
  --model_name_or_path adamlin/recipe-tag-model \
  --dataset_name recipe_dataset.py \
  --output_dir ./result_v2/ \
  --do_predict \
  --overwrite_output_dir \
  --num_train_epochs 1
```

6. run script
```bash=
python generate_data.py [cd data_tag/]
vim recipe_dataset.py [change features label]
bash run.sh
```
7. ssh login
  + ssh -i media1129-key-pair.cer ubuntu@34.204.168.125
  + tmux 
  + conda activate python3
  + pip uninstall transfoerms
  + pip install transformers
  + du -h
  + df -h -> /dev/xvda1

