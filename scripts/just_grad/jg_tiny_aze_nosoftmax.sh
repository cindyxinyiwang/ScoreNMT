#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --mem=18g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

#DDIR=/home/hyhieu/xinyiw/ScoreNMT/
#DDIR=/home/xinyiw1/ScoreNMT/
#DDIR=/home/hyhieu/xinyiw/ScoreNMT/
DDIR=/home/xinyiw/multv-nmt/

python3.6 src/rl_main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_s1/jg_tiny_aze_nosoftmax/" \
  --train_src_file_list "$DDIR"data/LAN_eng/ted-train.mtok.spm8000.LAN \
  --train_trg_file_list  "$DDIR"data_moses/LAN_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file_list  "$DDIR"data/aze_eng/ted-dev.mtok.spm8000.aze \
  --dev_trg_file_list  "$DDIR"data_moses/aze_eng/ted-dev.mtok.spm8000.eng \
  --dev_ref_file_list  "$DDIR"data_moses/aze_eng/ted-dev.mtok.eng \
  --dev_file_idx_list  "0" \
  --src_vocab_list  "$DDIR"data/LAN_eng/ted-train.mtok.spm8000.LAN.vocab \
  --src_char_vocab_from  "$DDIR"data/LAN_eng/ted-train.mtok.LAN.char4vocab \
  --char_ngram_n 4 \
  --src_char_vocab_size="8000" \
  --trg_vocab  "$DDIR"data_moses/eng/ted-train.mtok.spm8000.eng.vocab \
  --lang_file langs_tiny.txt \
  --lan_dist_file "$DDIR"ted-train-vocab.mtok.sim-ngram.graph \
  --base_lan "aze" \
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
  --lr_q 0.0001 \
  --refresh_base_grad=0 \
  --refresh_all_grad=1 \
  --n_train_epochs=20 \
  --dropout 0.3 \
  --max_len 250 \
  --agent_subsample_line 160 \
  --train_score_episode=100 \
  --print_every 50 \
  --init_train_score_every=5000 \
  --init_load_time=5 \
  --train_score_every=20000 \
  --record_grad_step=100 \
  --data_name="tiny" \
  --d_hidden 32 \
  --adam_raw_grad=0 \
  --imitate_episode 0 \
  --actor_type="spl" \
  --imitate_type "" \
  --feature_type="zero_one" \
  --norm_feature \
  --add_bias=0 \
  --just_grad \
  --cuda \
  --softmax_action 0 \
  --seed 1
