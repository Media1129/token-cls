# Recipe entity recognition task

## Generate recipe entity data
1. data folder -> data_tag/ [move slot_tagging data/slot/*.json file to this data folder]
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

6. predict.sh 
```bash=
python3 run_ner_backup.py \
  --model_name_or_path result_v2/checkpoint-3000 \
  --dataset_name recipe_dataset.py \
  --output_dir ./result_output_v2/ \
  --do_predict \
  --overwrite_output_dir \
  --num_train_epochs 1
```


7. run script
```bash=
python generate_data.py [cd data_tag/]
vim recipe_dataset.py [change model folder config.json]
bash run.sh
```
8. ssh login
  + ssh -i media1129-key-pair.cer ubuntu@34.204.168.125
  + tmux 
  + conda activate python3
  + pip uninstall transfoerms
  + pip install transformers
  + du -h
  + df -h -> /dev/xvda1

9. load ec2 model
scp -i media1129-key-pair.cer -r ubuntu@34.204.168.125:/home/ubuntu/Documents/token-cls/result_v2 /Users/media1129/Desktop/Alexa/token-cls/result_v2/

10. push model to huggingface hub
```bash=
cd ~/Desktop/recipe-tag-model
transformers-cli login
<!-- transformers-cli repo create recipe-tag-model -->
<!-- git clone https://huggingface.co/Media1129/recipe-tag-model -->

```

11. pipeline.py
  + pipeline.py [version1]
  + pipeline_v2.py [version2]
```bash=
python pipeline_v2.py
```
  
3000
  B-dishname I-dishname O O O O O O O
  2 O O O O B-ingredient O B-dishname
  3 O O O O B-dishname O O O O O
  4 O O O O O B-dishname I-dishname
  5 O O O O B-dishname I-dishname


6000
  B-dishname I-dishname O O O O O O B-dishname
  2 O O O O O O B-dishname
  3 O O O O B-dishname O O O O O
  4 O O O O O B-dishname I-dishname
  5 O O O O B-dishname I-dishname

all
  B-dishname I-dishname O O O O O O B-dishname
  2 O O O O O O B-dishname
  3 O O O O B-dishname O O O O O
  4 O O O O O B-dishname I-dishname
  5 O O O O B-dishname I-dishname
