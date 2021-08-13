python3 run_ner_backup.py \
  --model_name_or_path result_v2/checkpoint-6000 \
  --dataset_name recipe_dataset.py \
  --output_dir ./result_output_v2/ \
  --do_predict \
  --overwrite_output_dir \
  --num_train_epochs 1