export BASE_MODEL=allenai/scibert_scivocab_uncased
export OUTPUT_MODEL=./pretrain/output_models

python3 run_language_modeling.py \
	--output_dir ./pretrain \
	--train_data_file ./pretrain/proc_dataset.txt \
	--model_type bert \
	--model_name_or_path $BASE_MODEL \
	--mlm \
	--config_name ./pretrain \
	--tokenizer_name ./pretrain \
	--do_train \
	--line_by_line \
	--learning_rate 1e-4 \
	--num_train_epochs 1 \
	--save_total_limit 2 \
	--save_steps 2000 \
	--per_gpu_train_batch_size 4 \
	--seed 42 \
