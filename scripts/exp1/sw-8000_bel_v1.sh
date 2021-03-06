#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --mem=18g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

DDIR=/home/hyhieu/xinyiw1/ScoreNMT/

python3.6 src/rl_main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_exp1/sw-8000_bel_v1/" \
  --train_src_file_list "$DDIR"data/LAN_eng/ted-train.mtok.spm8000.LAN \
  --train_trg_file_list  "$DDIR"data_moses/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  "$DDIR"data/bel_eng/ted-dev.mtok.spm8000.bel \
  --dev_trg_file_list  "$DDIR"data_moses/bel_eng/ted-dev.mtok.spm8000.eng \
  --dev_ref_file_list  "$DDIR"data_moses/bel_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_vocab_list  "$DDIR"data/LAN_eng/ted-train.mtok.spm8000.LAN.vocab \
  --src_char_vocab_from  "$DDIR"data/LAN_eng/ted-train.mtok.LAN.char4vocab \
  --char_ngram_n 4 \
  --src_char_vocab_size="8000" \
  --trg_vocab  "$DDIR"data_moses/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file langs_tiny.txt \
  --lan_dist_file "$DDIR"ted-train-vocab.mtok.sim-ngram.graph \
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
  --max_len 380 \
  --agent_subsample_line 160 \
  --print_every 50 \
  --train_score_every=1978056 \
  --record_grad_step=0 \
  --data_name="tiny" \
  --output_prob_file="data/tiny/bel_exp1v6_" \
  --d_hidden 32 \
  --cuda \
  --imitate_episode 20 \
  --actor_type="base" \
  --not_train_score \
  --seed 0
