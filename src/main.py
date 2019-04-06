import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

from mult_data_utils import MultDataUtil
from hparams import *
from agent import *
from utils import *

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--always_save", action="store_true", help="always_save")

parser.add_argument("--semb", type=str, default=None, help="[mlp|dot_prod|linear]")
parser.add_argument("--dec_semb", action="store_true", help="load an existing model")
parser.add_argument("--query_base", action="store_true", help="load an existing model")
parser.add_argument("--semb_vsize", type=int, default=None, help="how many steps to write log")
parser.add_argument("--sep_char_proj", action="store_true", help="whether to have separate matrix for projecting char embedding")
parser.add_argument("--sep_relative_loc", action="store_true", help="whether to have separate transformer relative loc")
parser.add_argument("--residue", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--layer_norm", action="store_true", help="whether to set all unk words of rl to a reserved id")
parser.add_argument("--src_no_char", action="store_true", help="load an existing model")
parser.add_argument("--trg_no_char", action="store_true", help="load an existing model")
parser.add_argument("--char_gate", action="store_true", help="load an existing model")
parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
parser.add_argument("--lang_shuffle", action="store_true", help="load an existing model")
parser.add_argument("--ordered_char_dict", action="store_true", help="load an existing model")
parser.add_argument("--out_c_list", type=str, default=None, help="list of output channels for char cnn emb")
parser.add_argument("--k_list", type=str, default=None, help="list of kernel size for char cnn emb")
parser.add_argument("--highway", action="store_true", help="load an existing model")
parser.add_argument("--n", type=int, default=4, help="ngram n")
parser.add_argument("--single_n", action="store_true", help="ngram n")
parser.add_argument("--bpe_ngram", action="store_true", help="bpe ngram")
parser.add_argument("--uni", action="store_true", help="Gu Universal NMT")
parser.add_argument("--pretrained_src_emb_list", type=str, default=None, help="ngram n")
parser.add_argument("--pretrained_trg_emb", type=str, default=None, help="ngram n")

parser.add_argument("--load_model", action="store_true", help="load an existing model")
parser.add_argument("--reset_output_dir", action="store_true", help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs", help="path to output directory")
parser.add_argument("--log_every", type=int, default=50, help="how many steps to write log")
parser.add_argument("--eval_every", type=int, default=500, help="how many steps to compute valid ppl")
parser.add_argument("--clean_mem_every", type=int, default=10, help="how many steps to clean memory")
parser.add_argument("--eval_bleu", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for dev BLEU")
parser.add_argument("--poly_norm_m", type=float, default=1, help="beam size for dev BLEU")
parser.add_argument("--ppl_thresh", type=float, default=20, help="beam size for dev BLEU")
parser.add_argument("--max_trans_len", type=int, default=300, help="beam size for dev BLEU")
parser.add_argument("--merge_bpe", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--dev_zero", action="store_true", help="if eval at step 0")

parser.add_argument("--cuda", action="store_true", help="GPU or not")
parser.add_argument("--decode", action="store_true", help="whether to decode only")

parser.add_argument("--max_len", type=int, default=10000, help="maximum len considered on the target side")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")

parser.add_argument("--d_word_vec", type=int, default=288, help="size of word and positional embeddings")
parser.add_argument("--d_char_vec", type=int, default=None, help="size of word and positional embeddings")
parser.add_argument("--d_model", type=int, default=288, help="size of hidden states")
parser.add_argument("--d_inner", type=int, default=512, help="hidden dim of position-wise ff")
parser.add_argument("--n_layers", type=int, default=1, help="number of lstm layers")
parser.add_argument("--n_heads", type=int, default=3, help="number of attention heads")
parser.add_argument("--d_k", type=int, default=64, help="size of attention head")
parser.add_argument("--d_v", type=int, default=64, help="size of attention head")
parser.add_argument("--pos_emb_size", type=int, default=None, help="size of trainable pos emb")

parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--train_src_file_list", type=str, default=None, help="source train file")
parser.add_argument("--train_trg_file_list", type=str, default=None, help="target train file")
parser.add_argument("--dev_src_file_list", type=str, default=None, help="source valid file")
parser.add_argument("--dev_src_file", type=str, default=None, help="source valid file")
parser.add_argument("--dev_trg_file_list", type=str, default=None, help="target valid file")
parser.add_argument("--dev_trg_file", type=str, default=None, help="target valid file")
parser.add_argument("--dev_ref_file_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_trg_ref", type=str, default=None, help="target valid file for reference")
parser.add_argument("--dev_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--src_vocab_list", type=str, default=None, help="source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="target vocab file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="source test file")
parser.add_argument("--test_src_file", type=str, default=None, help="source test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="target test file")
parser.add_argument("--test_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--test_trg_file", type=str, default=None, help="target test file")
parser.add_argument("--src_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_vocab_size", type=int, default=None, help="src vocab size")
parser.add_argument("--trg_vocab_size", type=int, default=None, help="trg vocab size")
# multi data util options
parser.add_argument("--lang_file", type=str, default=None, help="language code file")
parser.add_argument("--src_vocab", type=str, default=None, help="source vocab file")
parser.add_argument("--src_vocab_from", type=str, default=None, help="list of source vocab file")
parser.add_argument("--trg_vocab", type=str, default=None, help="source vocab file")

parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--valid_batch_size", type=int, default=20, help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
parser.add_argument("--n_train_epochs", type=int, default=0, help="n_train_epochs")
parser.add_argument("--dropout", type=float, default=0., help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_q", type=float, default=0.001, help="learning rate for q function")
parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--lr_min", type=float, default=0.0001, help="min learning rate")
parser.add_argument("--lr_max", type=float, default=0.001, help="max learning rate")
parser.add_argument("--lr_dec_steps", type=int, default=0, help="cosine delay: learning rate decay steps")

parser.add_argument("--n_warm_ups", type=int, default=0, help="lr warm up steps")
parser.add_argument("--lr_schedule", action="store_true", help="whether to use transformer lr schedule")
parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
parser.add_argument("--l2_reg", type=float, default=0., help="L2 regularization")
parser.add_argument("--patience", type=int, default=-1, help="patience")
parser.add_argument("--eval_end_epoch", action="store_true", help="whether to reload the hparams")

parser.add_argument("--seed", type=int, default=19920206, help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

parser.add_argument("--share_emb_softmax", action="store_true", help="weight tieing")
parser.add_argument("--label_smoothing", type=float, default=None, help="label smooth")
parser.add_argument("--reset_hparams", action="store_true", help="whether to reload the hparams")

parser.add_argument("--char_ngram_n", type=int, default=0, help="use char_ngram embedding")
parser.add_argument("--max_char_vocab_size", type=int, default=None, help="char vocab size")

parser.add_argument("--char_input", type=str, default=None, help="[sum|cnn]")
parser.add_argument("--char_comb", type=str, default="add", help="[cat|add]")

parser.add_argument("--char_temp", type=float, default=None, help="temperature to combine word and char emb")

parser.add_argument("--pretrained_model", type=str, default=None, help="location of pretrained model")

parser.add_argument("--src_char_only", action="store_true", help="only use char emb on src")
parser.add_argument("--trg_char_only", action="store_true", help="only use char emb on trg")

parser.add_argument("--model_type", type=str, default="seq2seq", help="[seq2seq|transformer]")
parser.add_argument("--share_emb_and_softmax", action="store_true", help="only use char emb on trg")
parser.add_argument("--transformer_wdrop", action="store_true", help="whether to drop out word embedding of transformer")
parser.add_argument("--transformer_relative_pos", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_c", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--relative_pos_d", action="store_true", help="whether to use relative positional encoding of transformer")
parser.add_argument("--update_batch", type=int, default="1", help="for how many batches to call backward and optimizer update")
parser.add_argument("--layernorm_eps", type=float, default=1e-9, help="layernorm eps")

parser.add_argument("--num_data_feature", type=int, default="1", help="number of features for a batch of data")
parser.add_argument("--num_model_feature", type=int, default="1", help="number of features for the model")
parser.add_argument("--num_language_feature", type=int, default="1", help="number of features for the language")

parser.add_argument("--agent_dev_num", type=int, default="100", help="number of maximum dev sentences for evaluate the agent")

parser.add_argument("--lan_dist_file", type=str, default="", help="language distance file")
parser.add_argument("--base_lan", type=str, default="", help="language of interest")
parser.add_argument("--agent_subsample_percent", type=float, default=0, help="percentage of data subsampled to train the agent")
parser.add_argument("--agent_subsample_line", type=int, default=0, help="line count of data subsampled to train the agent")
parser.add_argument("--subsample", action="store_true", help="whether to subsample")
parser.add_argument("--update_agent_every", type=int, default=2, help="update data selection agent every")

parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
parser.add_argument("--replay_mem_size", type=int, default=64, help="replay memory size")
parser.add_argument("--replay_batch_size", type=int, default=16, help="replay batch size")
parser.add_argument("--sync_target_every", type=int, default=16, help="sync target Q network every")
parser.add_argument("--epsilon_max", type=float, default=0.05, help="epsilon greedy")
parser.add_argument("--epsilon_min", type=float, default=0.05, help="epsilon greedy")
parser.add_argument("--epsilon_anneal_step", type=float, default=10000, help="epsilon anneal steps")

parser.add_argument("--burn_in_size", type=int, default=3, help="number of items init in replay mem")
parser.add_argument("--test_step", type=int, default=30, help="number of steps for a episode of testing current policy")
parser.add_argument("--test_every", type=int, default=400, help="number of steps before testing current policy")
parser.add_argument("--print_every", type=int, default=50, help="log frequency")
parser.add_argument("--train_q_epoch", type=int, default=1, help="epochs to train q function per iteration")
parser.add_argument("--train_model_epoch", type=int, default=2, help="epochs to train model per iteration")
parser.add_argument("--train_model_step", type=int, default=100000, help="steps to train model per iteration")
parser.add_argument("--max_iter", type=int, default=10, help="maximum training iteration")
parser.add_argument("--double_q", action="store_true", help="whether to use double q learning")
parser.add_argument("--add_model_feature", action="store_true", help="whether to use model feature")
parser.add_argument("--train_q_every", type=int, default=0, help="train q for how many steps of model training")

parser.add_argument("--balance_sample", action="store_true", help="balance sample of replay mem")
parser.add_argument("--src_static_feature_only", action="store_true", help="only use the static src feature")

parser.add_argument("--neg_r", type=float, default=-2, help="neg reward")
parser.add_argument("--pos_r", type=float, default=2, help="neg reward")
parser.add_argument("--eqn_r", type=float, default=0, help="neg reward")

parser.add_argument("--train_file_idx", type=str, default="", help="if set, train using the given files")
parser.add_argument("--reset_nmt", type=int, default=1, help="whether to reset nmt state after one episode of q training")
parser.add_argument("--pretrain_nmt_epoch", type=int, default=0, help="whether to train nmt before dqn")
parser.add_argument("--add_dev_logit_feature", type=int, default=0, help="whether to features from dev logits")

parser.add_argument("--d_hidden", type=int, default=128, help="dimension of hidden of Q network")
parser.add_argument("--reward_type", type=str, default="fixed", help="[fixed|dynamic]")
args = parser.parse_args()

def train():
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps
  else:
    hparams = HParams(
      decode=args.decode,
      data_path=args.data_path,
      train_src_file_list=args.train_src_file_list,
      train_trg_file_list=args.train_trg_file_list,
      dev_src_file=args.dev_src_file,
      dev_trg_file=args.dev_trg_file,
      dev_src_file_list=args.dev_src_file_list,
      dev_trg_file_list=args.dev_trg_file_list,
      dev_ref_file_list=args.dev_ref_file_list,
      dev_file_idx_list=args.dev_file_idx_list,
      lang_file=args.lang_file,
      src_vocab=args.src_vocab,
      trg_vocab=args.trg_vocab,
      src_vocab_list=args.src_vocab_list,
      trg_vocab_list=args.trg_vocab_list,
      src_vocab_size=args.src_vocab_size,
      trg_vocab_size=args.trg_vocab_size,
      max_len=args.max_len,
      n_train_sents=args.n_train_sents,
      cuda=args.cuda,
      clean_mem_every=args.clean_mem_every,
      d_word_vec=args.d_word_vec,
      d_model=args.d_model,
      d_inner=args.d_inner,
      n_layers=args.n_layers,
      batch_size=args.batch_size,
      valid_batch_size=args.valid_batch_size,
      batcher=args.batcher,
      n_train_steps=args.n_train_steps,
      dropout=args.dropout,
      lr=args.lr,
      lr_q=args.lr_q,
      patience=args.patience,
      log_every=args.log_every,
      clip_grad=args.clip_grad,
      lr_dec=args.lr_dec,
      l2_reg=args.l2_reg,
      init_type=args.init_type,
      init_range=args.init_range,
      share_emb_softmax=args.share_emb_softmax,
      label_smoothing=args.label_smoothing,
      n_heads=args.n_heads,
      d_k=args.d_k,
      d_v=args.d_v,
      merge_bpe=args.merge_bpe,
      load_model=args.load_model,
      char_ngram_n=args.char_ngram_n,
      max_char_vocab_size=args.max_char_vocab_size,
      char_input=args.char_input,
      char_comb=args.char_comb,
      char_temp=args.char_temp,
      src_char_vocab_from=args.src_char_vocab_from,
      src_char_vocab_size=args.src_char_vocab_size,
      trg_char_vocab_from=args.trg_char_vocab_from,
      trg_char_vocab_size=args.trg_char_vocab_size,
      src_char_only=args.src_char_only,
      trg_char_only=args.trg_char_only,
      semb=args.semb,
      dec_semb=args.dec_semb,
      semb_vsize=args.semb_vsize,
      query_base=args.query_base,
      residue=args.residue,
      layer_norm=args.layer_norm,
      src_no_char=args.src_no_char,
      trg_no_char=args.trg_no_char,
      char_gate=args.char_gate,
      shuffle_train=args.shuffle_train,
      lang_shuffle=args.lang_shuffle,
      ordered_char_dict=args.ordered_char_dict,
      out_c_list=args.out_c_list,
      k_list=args.k_list,
      d_char_vec=args.d_char_vec,
      highway=args.highway,
      n=args.n,
      single_n=args.single_n,
      bpe_ngram=args.bpe_ngram,
      uni=args.uni,
      pos_emb_size=args.pos_emb_size,
      lr_schedule=args.lr_schedule,
      lr_max=args.lr_max,
      lr_min=args.lr_min,
      lr_dec_steps=args.lr_dec_steps,
      n_warm_ups=args.n_warm_ups,
      model_type=args.model_type,
      transformer_wdrop=args.transformer_wdrop,
      transformer_relative_pos=args.transformer_relative_pos,
      relative_pos_c=args.relative_pos_c,
      relative_pos_d=args.relative_pos_d,
      update_batch=args.update_batch,
      num_data_feature=args.num_data_feature,
      num_model_feature=args.num_model_feature,
      num_language_feature=args.num_language_feature,
      agent_dev_num=args.agent_dev_num,
      lan_dist_file=args.lan_dist_file,
      base_lan=args.base_lan,
      agent_subsample_percent=args.agent_subsample_percent,
      agent_subsample_line=args.agent_subsample_line,
      subsample=args.subsample,
      update_agent_every=args.update_agent_every,
      gamma=args.gamma,
      replay_mem_size=args.replay_mem_size,
      replay_batch_size=args.replay_batch_size,
      sync_target_every=args.sync_target_every,
      burn_in_size=args.burn_in_size,
      epsilon_min=args.epsilon_min,
      epsilon_max=args.epsilon_max,
      epsilon_anneal_step=args.epsilon_anneal_step,
      test_step=args.test_step,
      test_every=args.test_every,
      print_every=args.print_every,
      train_q_epoch=args.train_q_epoch,
      train_model_epoch=args.train_model_epoch,
      train_model_step=args.train_model_step,
      max_iter=args.max_iter,

      eval_every=args.eval_every,
      output_dir=args.output_dir,
      double_q=args.double_q,
      add_model_feature=args.add_model_feature,
      train_q_every=args.train_q_every,
      balance_sample=args.balance_sample,
      src_static_feature_only=args.src_static_feature_only,
      neg_r=args.neg_r,
      pos_r=args.pos_r,
      eqn_r=args.eqn_r,
      train_file_idx=args.train_file_idx,
      reset_nmt=args.reset_nmt,
      pretrain_nmt_epoch=args.pretrain_nmt_epoch,
      d_hidden=args.d_hidden,
      add_dev_logit_feature=args.add_dev_logit_feature,
      reward_type=args.reward_type,
    )
  # build or load model
  print("-" * 80)
  print("Creating model")
  agent_trainer = AgentTrainer(hparams)
  agent_trainer.train()

def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("-" * 80)
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

  print("-" * 80)
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()

if __name__ == "__main__":
  main()
