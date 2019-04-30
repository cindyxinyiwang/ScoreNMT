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

from rl_data_utils import RLDataUtil
from hparams import *
from reinforce import *
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
parser.add_argument("--test_ref_file_list", type=str, default=None, help="target test file")
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

parser.add_argument("--raw_batch_size", type=int, default=1, help="batch_size for raw examples for RL")
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
parser.add_argument("--actor_init_range", type=float, default=0.1, help="L2 init range")
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

parser.add_argument("--burn_in_size", type=int, default=100, help="number of items init in replay mem")
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
parser.add_argument("--data_name", type=str, default="tiny", help="dir name of processed data")
parser.add_argument("--episode", type=int, default=10, help="number of episode")

parser.add_argument("--min_temp", type=float, default=0.1, help="min temp for sampling action")
parser.add_argument("--max_temp", type=float, default=0.9, help="max temp for sampling action")
parser.add_argument("--delta_temp", type=float, default=0.001, help="max temp for sampling action")

parser.add_argument("--output_prob_file", type=str, default="", help="file to write the output probs")
parser.add_argument("--nmt_train", action="store_true", help="train nmt")
parser.add_argument("--nmt_train_prob_lr", action="store_true", help="train nmt")
parser.add_argument("--detok", action="store_true", help="whether to detokenize data")
parser.add_argument("--min_lr_prob", type=float, default=0, help="min lr prob to use a data")
parser.add_argument("--bias", type=float, default=1.0, help="bias of the language distance")
parser.add_argument("--iteration", type=int, default=1, help="number of times to train the RL agent")
parser.add_argument("--random_rl_train", type=int, default=0, help="whether to randomize rl training data feed")

parser.add_argument("--discard_bias", type=int, default=-100, help="bias term to discard a sentence")
parser.add_argument("--feature_type", type=str, default="lan_dist", help="[lan_dist|zero_one]")

parser.add_argument("--actor_type", type=str, default="base", help="[base|emb]")
parser.add_argument("--add_bias", type=int, default=1, help="whether to add bias to actor logits")
parser.add_argument("--imitate_episode", type=int, default=0, help="episodes to imitate")

parser.add_argument("--train_score_every", type=int, default=0, help="episodes to imitate")
parser.add_argument("--record_grad_step", type=int, default=50, help="episodes to imitate")

parser.add_argument("--not_train_score", action="store_true", help="do not train the score function")
parser.add_argument("--train_score_episode", type=int, default=1, help="how many updates to train the score")
parser.add_argument("--refresh_base_grad", type=int, default=1, help="whether to refresh the grad on base lan before updating score")
parser.add_argument("--refresh_all_grad", type=int, default=0, help="whether to refresh the grad on all lan before updating score")
parser.add_argument("--refresh_num", type=int, default=30, help="the number of sentences to refresh")

parser.add_argument("--model_optimizer", type=str, default="ADAM", help="[SGD|ADAM]")

parser.add_argument("--cosine_schedule_max_step", type=int, default=0, help="the max step for cosine lr anneal")
parser.add_argument("--schedule_restart", type=int, default=1, help="[1|0] whether to restart schedule at eop")

parser.add_argument("--init_load_time", type=int, default=0, help="number of times to use init train score every")
parser.add_argument("--init_train_score_every", type=int, default=1000, help="initial train score every")

parser.add_argument("--scale_0", type=float, default=0.05, help="scale for past gradient")
parser.add_argument("--scale_1", type=float, default=0.95, help="scale for current gradient")

parser.add_argument("--adam_raw_grad", type=int, default=0, help="[0|1] whether to use raw grad for adam")
parser.add_argument("--agent_checkpoint_every", type=int, default=20, help="save agent every n epoch")

parser.add_argument("--grad_dist", type=str, default="cosine", help="[cosine|dot_prod]")
parser.add_argument("--reward_scale", type=float, default=1, help="scale of the gradient sim")
parser.add_argument("--imitate_type", type=str, default="heuristic", help="[heuristic|init]")

parser.add_argument("--bucketed", action="store_true", help="whether to use bucketed version of RL")
parser.add_argument("--norm_feature", action="store_true", help="whether to normalize the feature for actor")
parser.add_argument("--sample_all", action="store_true", help="whether to sample all languages for a given target")

parser.add_argument("--baseline", action="store_true", help="whether to use baseline for grad prod")
parser.add_argument("--baseline_scale_0", type=float, default=0.2, help="weight for prev grad prod")
parser.add_argument("--baseline_scale_1", type=float, default=0.8, help="weight for current grad prod")

parser.add_argument("--reverse_sign", action="store_true", help="whether to reverse the sign")

parser.add_argument("--lrl_loss", type=float, default=0., help="weight for lrl loss")
parser.add_argument("--mask_loss", type=int, default=1, help="whether to reverse the sign")
parser.add_argument("--norm_bucket_instance", action="store_true", help="whether to reverse the sign")
parser.add_argument("--shuffle_bucket", type=int, default=1, help="whether to shuffle data in each bucket")
parser.add_argument("--pretrained_actor", type=str, default="", help="whether to shuffle data in each bucket")
args = parser.parse_args()

def train():
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps
  else:
    hparams = HParams(**vars(args))
  # build or load model
  if args.nmt_train:
    train_nmt(hparams)
  elif args.nmt_train_prob_lr:
    train_nmt_prob_lr(hparams)
  elif args.decode:
    agent_trainer = ReinforceTrainer(hparams)
    print("-" * 80)
    print("Decoding")
    agent_trainer.decode()
  else:
    agent_trainer = ReinforceTrainer(hparams)
    print("-" * 80)
    print("Creating model")
    #agent_trainer.train()
    agent_trainer.train_rl_and_nmt()

def train_nmt_prob_lr(hparams):
  data_util = RLDataUtil(hparams)
  model = Seq2Seq(hparams=hparams, data=data_util)
  print("initialize uniform with range {}".format(args.init_range))
  for p in model.parameters():
    p.data.uniform_(-args.init_range, args.init_range)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
  #optim = torch.optim.Adam(trainable_params)
  step = 0
  best_val_ppl = [None for _ in range(len(model.hparams.dev_src_file_list))]
  best_val_bleu = [None for _ in range(len(model.hparams.dev_src_file_list))]
  cur_attempt = 0
  lr = hparams.lr

  if args.cuda:
    model = model.cuda()

  if args.reset_hparams:
    lr = args.lr
  crit = get_criterion(hparams)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  target_words, total_loss, total_corrects = 0, 0, 0
  target_rules, target_total, target_eos = 0, 0, 0
  total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
  model.train()
  #i = 0
  epoch = 0
  tr_loss, update_batch_size = None, 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, prob, batch_size, eop) in data_util.next_nmt_train_prob_lr():
    step += 1
    target_words += (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, [], [], file_idx=[], step=step, x_rank=[])
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, hparams, batch_size=batch_size, element_weight=prob)
    total_loss += cur_tr_loss.item()
    total_corrects += cur_tr_acc.item()
    cur_tr_loss.div_(prob.sum().item() * hparams.update_batch)

    cur_tr_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    if tr_loss is None:
      tr_loss = cur_tr_loss.item()
    else:
      tr_loss = tr_loss + cur_tr_loss.item()

    if step % args.update_batch == 0:
      optim.step()
      optim.zero_grad()
      tr_loss = None
      update_batch_size = 0
    # clean up GPU memory
    if step % args.clean_mem_every == 0:
      gc.collect()
    if eop: 
      epoch += 1
    #  get_grad_cos_all(model, data, crit)
    if (step / args.update_batch) % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step / args.update_batch) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(cur_tr_loss.item())
      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
      log_string += " acc={0:<5.4f}".format(total_corrects / target_words)

      log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
    if args.eval_end_epoch:
      if eop:
        eval_now = True
      else:
        eval_now = False
    elif (step / args.update_batch) % args.eval_every == 0:
      eval_now = True
    else:
      eval_now = False 
    if eval_now:
      based_on_bleu = args.eval_bleu and best_val_ppl[0] is not None and best_val_ppl[0] <= args.ppl_thresh
      if args.dev_zero: based_on_bleu = True
      with torch.no_grad():
        val_ppl, val_bleu, ppl_list, bleu_list = eval(model, data_util, crit, step, hparams, args, eval_bleu=based_on_bleu, valid_batch_size=args.valid_batch_size, tr_logits=logits)	
      for i in range(len(ppl_list)):
        if based_on_bleu:
          if best_val_bleu[i] is None or best_val_bleu[i] <= bleu_list[i]:
            save = True 
            best_val_bleu[i] = bleu_list[i]
            cur_attempt = 0
          else:
            save = False
            cur_attempt += 1
        else:
       	  if best_val_ppl[i] is None or best_val_ppl[i] >= ppl_list[i]:
            save = True
            best_val_ppl[i] = ppl_list[i]
            cur_attempt = 0 
          else:
            save = False
            cur_attempt += 1
        if save or args.always_save:
          if len(ppl_list) > 1:
            nmt_save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir + "dev{}".format(i))
          else:
            nmt_save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir)
        elif not args.lr_schedule and step >= hparams.n_warm_ups:
          lr = lr * args.lr_dec
          set_lr(optim, lr)
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_corrects = total_loss = 0
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
    if args.patience >= 0:
      if cur_attempt > args.patience: break
    elif args.n_train_epochs > 0:
      if epoch >= args.n_train_epochs: break
    else:
      if step > args.n_train_steps: break


def train_nmt(hparams):
  data_util = RLDataUtil(hparams)
  model = Seq2Seq(hparams=hparams, data=data_util)
  print("initialize uniform with range {}".format(args.init_range))
  for p in model.parameters():
    p.data.uniform_(-args.init_range, args.init_range)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
  #optim = torch.optim.Adam(trainable_params)
  step = 0
  best_val_ppl = [None for _ in range(len(model.hparams.dev_src_file_list))]
  best_val_bleu = [None for _ in range(len(model.hparams.dev_src_file_list))]
  cur_attempt = 0
  lr = hparams.lr

  if args.cuda:
    model = model.cuda()

  if args.reset_hparams:
    lr = args.lr
  crit = get_criterion(hparams)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  target_words, total_loss, total_corrects = 0, 0, 0
  target_rules, target_total, target_eos = 0, 0, 0
  total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
  model.train()
  #i = 0
  epoch = 0
  tr_loss, update_batch_size = None, 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, eop) in data_util.next_nmt_train():
    step += 1
    target_words += (y_count - batch_size)
    logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, [], [], file_idx=[], step=step, x_rank=[])
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y_train[:,1:].contiguous().view(-1)
      
    cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, hparams)
    total_loss += cur_tr_loss.item()
    total_corrects += cur_tr_acc.item()
    cur_tr_loss.div_(batch_size * hparams.update_batch)

    cur_tr_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    if tr_loss is None:
      tr_loss = cur_tr_loss.item()
    else:
      tr_loss = tr_loss + cur_tr_loss.item()

    if step % args.update_batch == 0:
      optim.step()
      optim.zero_grad()
      tr_loss = None
      update_batch_size = 0
    # clean up GPU memory
    if step % args.clean_mem_every == 0:
      gc.collect()
    if eop: 
      epoch += 1
    #  get_grad_cos_all(model, data, crit)
    if (step / args.update_batch) % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step / args.update_batch) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(cur_tr_loss.item())
      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
      log_string += " acc={0:<5.4f}".format(total_corrects / target_words)

      log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
    if args.eval_end_epoch:
      if eop:
        eval_now = True
      else:
        eval_now = False
    elif (step / args.update_batch) % args.eval_every == 0:
      eval_now = True
    else:
      eval_now = False 
    if eval_now:
      based_on_bleu = args.eval_bleu and best_val_ppl[0] is not None and best_val_ppl[0] <= args.ppl_thresh
      if args.dev_zero: based_on_bleu = True
      with torch.no_grad():
        val_ppl, val_bleu, ppl_list, bleu_list = eval(model, data_util, crit, step, hparams, args, eval_bleu=based_on_bleu, valid_batch_size=args.valid_batch_size, tr_logits=logits)	
      for i in range(len(ppl_list)):
        if based_on_bleu:
          if best_val_bleu[i] is None or best_val_bleu[i] <= bleu_list[i]:
            save = True 
            best_val_bleu[i] = bleu_list[i]
            cur_attempt = 0
          else:
            save = False
            cur_attempt += 1
        else:
       	  if best_val_ppl[i] is None or best_val_ppl[i] >= ppl_list[i]:
            save = True
            best_val_ppl[i] = ppl_list[i]
            cur_attempt = 0 
          else:
            save = False
            cur_attempt += 1
        if save or args.always_save:
          if len(ppl_list) > 1:
            nmt_save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir + "dev{}".format(i))
          else:
            nmt_save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], model, optim, hparams, args.output_dir)
        elif not args.lr_schedule and step >= hparams.n_warm_ups:
          lr = lr * args.lr_dec
          set_lr(optim, lr)
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_corrects = total_loss = 0
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
    if args.patience >= 0:
      if cur_attempt > args.patience: break
    elif args.n_train_epochs > 0:
      if epoch >= args.n_train_epochs: break
    else:
      if step > args.n_train_steps: break


def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  #elif args.reset_output_dir:
  #  print("-" * 80)
  #  print("Path {} exists. Remove and remake.".format(args.output_dir))
  #  shutil.rmtree(args.output_dir)
  #  os.makedirs(args.output_dir)

  if args.nmt_train:
    print("-" * 80)
    log_file = os.path.join(args.output_dir, "nmt_stdout")
    print("Logging to {}".format(log_file))
    sys.stdout = Logger(log_file)
    train()
  elif args.decode:
    print("-" * 80)
    log_file = os.path.join(args.output_dir, "decode_stdout")
    print("Logging to {}".format(log_file))
    sys.stdout = Logger(log_file)
    train()
  else:
    print("-" * 80)
    log_file = os.path.join(args.output_dir, "stdout")
    print("Logging to {}".format(log_file))
    sys.stdout = Logger(log_file)
    train()

  # translate dev test
  trans_script = 'python3.6 src/translate.py \
  --model_file="{}/ppl_final_nmt_model.pt" \
  --hparams_file="{}/ppl_final_nmt_hparams.pt" \
  --test_src_file_list  {} \
  --test_trg_file_list  {} \
  --test_ref_file_list  {} \
  --test_file_idx_list "0" \
  --cuda \
  --merge_bpe \
  --beam_size=5 \
  --poly_norm_m=1 \
  --max_len=200 \
  --trans_dev \
  --log_file="{}/ppl_trans_log" \
  --out_prefix="ppl" \
  --out_file="{}/ppl-ted-test-b5m1"'.format(args.output_dir, args.output_dir, args.test_src_file_list, args.test_trg_file_list, args.test_ref_file_list, args.output_dir, args.output_dir)
  out = subprocess.getoutput(trans_script)
  print(out)
  trans_script = 'python3.6 src/translate.py \
  --model_file="{}/bleu_final_nmt_model.pt" \
  --hparams_file="{}/bleu_final_nmt_hparams.pt" \
  --test_src_file_list  {} \
  --test_trg_file_list  {} \
  --test_ref_file_list  {} \
  --test_file_idx_list "0" \
  --cuda \
  --merge_bpe \
  --beam_size=5 \
  --poly_norm_m=1 \
  --max_len=200 \
  --trans_dev \
  --log_file="{}/bleu_trans_log" \
  --out_prefix="bleu" \
  --out_file="{}/bleu-ted-test-b5m1"'.format(args.output_dir, args.output_dir, args.test_src_file_list, args.test_trg_file_list, args.test_ref_file_list, args.output_dir, args.output_dir)
  out = subprocess.getoutput(trans_script)
  print(out)
if __name__ == "__main__":
  main()
