from transformers import *

#from hugging

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')


#1. retrain vocab.txt 
#using https://github.com/google/sentencepiece
#pip install sentencepiece
import sentencepiece as spm
s = spm.SentencePieceProcessor()
s.Load('spm.model')
spm.SentencePieceTrainer.Train('--input=combined.out --model_prefix=100B_9999_cased --vocab_size=31000 \
								--character_coverage=0.9999 --model_type=bpe --input_sentence_size=100000000 \
								--shuffle_input_sentence=true')

#append the covid vocab.txt to the original vocab.txt from scibert 

#2. conver dataset into sentence pretrain o sentense 



#3. using BERT to pretrain 
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python3 run_pretraining.py \
	--input_file=/tmp/tf_examples.tfrecord \  #change to new PATH
	--output_dir=/tmp/pretraining_output \  #change to cloud PATH 
	--do_train=True \
	--do_eval=True \
	--bert_config_file=/mnt/disk1/bert_config/s2vocab_uncased.json \
	--train_batch_size=64 \
	--max_seq_length=512 \
	--max_predictions_per_seq=75 \
	--num_train_steps=800000 \
	--num_warmup_steps=100 \
	--learning_rate=1e-5 \
	#--use_tpu=True  \#change 
	#--tpu_name=node-1 \#change
	--max_eval_steps=2000  \
	--eval_batch_size 64 \
	#--init_checkpoint=gs://s2-bert/s2-models/3B-s2vocab_uncased_512_finetune128 
	#--tpu_zone=us-central1-a #

