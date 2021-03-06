#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --mem=18g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="3"

#DDIR=/home/hyhieu/xinyiw/ScoreNMT/
#DDIR=/home/xinyiw1/ScoreNMT/
#DDIR=/home/hyhieu/xinyiw/ScoreNMT/
DDIR=/home/xinyiw/data/

CUDA_VISIBLE_DEVICES=$1 python src/rl_main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_s1/spl_tiny_bel/" \
  --train_src_file_list "$DDIR"ted/LAN_eng/ted-train.mtok.spm8000.LAN \
  --train_trg_file_list  "$DDIR"ted_moses/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  "$DDIR"ted/bel_eng/ted-dev.mtok.spm8000.bel \
  --dev_trg_file_list  "$DDIR"ted_moses/bel_eng/ted-dev.mtok.spm8000.eng \
  --dev_ref_file_list  "$DDIR"ted_moses/bel_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_vocab_list  "$DDIR"ted/LAN_eng/ted-train.mtok.spm8000.LAN.vocab \
  --src_char_vocab_from  "$DDIR"ted/LAN_eng/ted-train.mtok.LAN.char4vocab \
  --char_ngram_n 4 \
  --src_char_vocab_size="8000" \
  --trg_vocab  "$DDIR"ted_moses/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file langs_tiny.txt \
  --lan_dist_file /home/xinyiw/multv-nmt/ted-train-vocab.mtok.sim-ngram.graph \
  --base_lan "bel" \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --batcher='word' \
  --batch_size 1500 \
  --raw_batch_size 1 \
  --lr_dec 1.0 \
  --lr 0.001 \
  --n_train_epochs=20 \
  --dropout 0.3 \
  --max_len 250 \
  --agent_subsample_line 160 \
  --print_every 50 \
  --train_score_every=50000 \
  --record_grad_step=0 \
  --data_name="tiny" \
  --d_hidden 32 \
  --cuda \
  --imitate_episode 0 \
  --imitate_type "" \
  --actor_type="spl" \
  --feature_type="zero_one" \
  --add_bias=0 \
  --not_train_score \
  --spl_actor \
  --seed 1
