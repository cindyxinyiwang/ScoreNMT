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

from data_utils import DataUtil
from mult_data_utils import MultDataUtil
from class_data_utils import ClassDataUtil
from hparams import *
from utils import *
from model import *

parser = argparse.ArgumentParser(description="classify")

parser.add_argument("--semb", type=str, default=None, help="[mlp|dot_prod|linear]")
parser.add_argument("--semb_vsize", type=int, default=None, help="how many steps to write log")
parser.add_argument("--trg_no_char", action="store_true", help="load an existing model")
parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
parser.add_argument("--ordered_char_dict", action="store_true", help="load an existing model")
parser.add_argument("--bpe_ngram", action="store_true", help="bpe ngram")

parser.add_argument("--load_model", action="store_true", help="load an existing model")
parser.add_argument("--reset_hparams", action="store_true", help="whether to reload the hparams")
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
parser.add_argument("--k_list", type=str, default="2,3,4", help="filter size")
parser.add_argument("--out_c_list", type=str, default="100,100,100", help="output channel")

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
parser.add_argument("--test_file_idx_list", type=str, default=None, help="target valid file for reference")
parser.add_argument("--src_vocab_list", type=str, default=None, help="source vocab file")
parser.add_argument("--trg_vocab_list", type=str, default=None, help="target vocab file")
parser.add_argument("--test_src_file_list", type=str, default=None, help="source test file")
parser.add_argument("--test_src_file", type=str, default=None, help="source test file")
parser.add_argument("--test_trg_file_list", type=str, default=None, help="target test file")
parser.add_argument("--test_trg_file", type=str, default=None, help="target test file")
parser.add_argument("--src_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_char_vocab_size", type=str, default=None, help="source char vocab file")
parser.add_argument("--trg_char_vocab_from", type=str, default=None, help="source char vocab file")
parser.add_argument("--src_vocab_size", type=int, default=None, help="src vocab size")
parser.add_argument("--trg_vocab_size", type=int, default=None, help="trg vocab size")
parser.add_argument("--char_ngram_n", type=int, default=0, help="use char_ngram embedding")
parser.add_argument("--max_char_vocab_size", type=int, default=None, help="char vocab size")
parser.add_argument("--sep_char_proj", action="store_true", help="if eval at step 0")

# multi data util options
parser.add_argument("--lang_file", type=str, default=None, help="language code file")
parser.add_argument("--test_lang_file", type=str, default=None, help="language code file")
parser.add_argument("--src_vocab", type=str, default=None, help="source vocab file")
parser.add_argument("--src_vocab_from", type=str, default=None, help="list of source vocab file")
parser.add_argument("--trg_vocab", type=str, default=None, help="source vocab file")

parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--valid_batch_size", type=int, default=20, help="batch_size")
parser.add_argument("--test_batch_size", type=int, default=20, help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
parser.add_argument("--n_train_epochs", type=int, default=0, help="n_train_epochs")
parser.add_argument("--dropout", type=float, default=0., help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
parser.add_argument("--l2_reg", type=float, default=0., help="L2 regularization")
parser.add_argument("--patience", type=int, default=-1, help="patience")

parser.add_argument("--seed", type=int, default=19920206, help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

args = parser.parse_args()


class CNNClassify(nn.Module):

  def __init__(self, hparams):
    super(CNNClassify, self).__init__()
    self.hparams = hparams
    if self.hparams.semb:
      print("using SDE...")
      if self.hparams.semb_vsize is None:
        self.hparams.semb_vsize = self.hparams.src_vocab_size 
      self.word_emb = QueryEmb(self.hparams, self.hparams.semb_vsize)
      self.char_emb = charEmbedder(self.hparams, char_vsize=self.hparams.src_char_vsize)
    else:
      self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                   self.hparams.d_word_vec,
                                   padding_idx=hparams.pad_id)

    self.conv_list = []
    self.mask_conv_list = []
    for c, k in zip(self.hparams.out_c_list, self.hparams.k_list):
      #self.conv_list.append(nn.Conv1d(self.hparams.d_word_vec, out_channels=c, kernel_size=k, padding = k // 2))
      self.conv_list.append(nn.Conv1d(self.hparams.d_word_vec, out_channels=c, kernel_size=k))
      nn.init.uniform_(self.conv_list[-1].weight, -args.init_range, args.init_range)
      self.mask_conv_list.append(nn.Conv1d(1, out_channels=c, kernel_size=k))
      nn.init.constant_(self.mask_conv_list[-1].weight, 1.0)

    self.conv_list = nn.ModuleList(self.conv_list)
    self.mask_conv_list = nn.ModuleList(self.mask_conv_list)
    for param in self.mask_conv_list.parameters():
      param.requires_grad = False

    self.project = nn.Linear(sum(self.hparams.out_c_list), self.hparams.trg_vocab_size, bias=False)
    nn.init.uniform_(self.project.weight, -args.init_range, args.init_range)
    if self.hparams.cuda:
      self.conv_list = self.conv_list.cuda()
      self.project = self.project.cuda()

  def forward(self, x_train, x_mask, x_len, x_train_char, file_idx=None, step=None):
    if x_train_char:
      batch_size, max_len = len(x_train_char), len(x_train_char[0])
    else:
      batch_size, max_len = x_train.size()

    # [batch_size, max_len, d_word_vec]
    if self.hparams.semb:
      char_emb = self.char_emb(x_train_char, file_idx=file_idx)
      word_emb = self.word_emb(char_emb, file_idx=file_idx)
      word_emb = word_emb
    else:
      word_emb = self.word_emb(x_train)

    #x_mask = x_mask.unsqueeze(1).float()
    # [batch_size, d_word_vec, max_len]
    word_emb = word_emb.permute(0, 2, 1)
    conv_out = []
    for conv, m_conv in zip(self.conv_list, self.mask_conv_list):
      # [batch_size, c_out, max_len]
      c = conv(word_emb)
      #with torch.no_grad():
      #  m = m_conv(x_mask)
      #print(m_conv.weight)
      #print(m)
      #m = (m > 0)
      #print(m)
      #c.masked_fill_(m, -float("inf"))
      # [batch_size, c_out]
      c = c.max(dim=-1)
      conv_out.append(c[0])
    # [batch_size, trg_vocab_size]
    logits = self.project(torch.cat(conv_out, dim=-1))
    return logits

def eval(model, data, crit, step, hparams):
  print("Eval at step {0}. valid_batch_size={1}".format(step, args.valid_batch_size))
  model.hparams.decode = True
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_acc, total_loss = 0, 0
  valid_bleu = None
  file_count = 0
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, batch_size, x_char, y_char, eop, eof, dev_file_index in data.next_dev(dev_batch_size=args.valid_batch_size):
    # clear GPU memory
    gc.collect()

    # next batch
    logits = model.forward(
      x, x_mask, x_len, x_char, file_idx=dev_file_index, step=step)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = Variable(torch.LongTensor(y))
    if args.cuda: labels = labels.cuda()
    val_loss = crit(logits, labels)
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()
    #print(labels)
    n_batches += batch_size
    valid_loss += val_loss.sum().item()
    valid_acc += val_acc.item()
    if eof:
      print("val_step={0:<6d}".format(step))
      print(" loss={0:<6.2f}".format(valid_loss / n_batches))
      print(" acc={0:<5.4f}".format(valid_acc / n_batches))
      total_loss += valid_loss
      total_acc += valid_acc
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      file_count += 1
    if eop:
      break
  return total_acc / file_count, total_loss

def train():
  if args.load_model and (not args.reset_hparams):
    print("load hparams..")
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.load_model = args.load_model
    hparams.n_train_steps = args.n_train_steps

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=hparams.l2_reg)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr, weight_decay=hparams.l2_reg)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, best_val_bleu, cur_attempt, lr = torch.load(extra_file_name)
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
      test_src_file_list=args.test_src_file_list,
      lang_file=args.lang_file,
      test_lang_file=args.test_lang_file,
      src_vocab=args.src_vocab,
      trg_vocab=args.trg_vocab,
      src_vocab_list=args.src_vocab_list,
      trg_vocab_list=args.trg_vocab_list,
      src_vocab_size=args.src_vocab_size,
      trg_vocab_size=args.trg_vocab_size,
      max_len=args.max_len,
      n_train_sents=args.n_train_sents,
      cuda=args.cuda,
      d_word_vec=args.d_word_vec,
      d_model=args.d_model,
      d_inner=args.d_inner,
      n_layers=args.n_layers,
      k_list=args.k_list,
      out_c_list=args.out_c_list,
      batch_size=args.batch_size,
      batcher=args.batcher,
      n_train_steps=args.n_train_steps,
      dropout=args.dropout,
      lr=args.lr,
      lr_dec=args.lr_dec,
      init_type=args.init_type,
      init_range=args.init_range,
      merge_bpe=args.merge_bpe,
      load_model=args.load_model,
      char_ngram_n=args.char_ngram_n,
      max_char_vocab_size=args.max_char_vocab_size,
      src_char_vocab_from=args.src_char_vocab_from,
      src_char_vocab_size=args.src_char_vocab_size,
      semb=args.semb,
      semb_vsize=args.semb_vsize,
      sep_char_proj=args.sep_char_proj,
      shuffle_train=args.shuffle_train,
      ordered_char_dict=args.ordered_char_dict,
      d_char_vec=args.d_char_vec,
      bpe_ngram=args.bpe_ngram,
    )
  print("building model...")
  if args.load_model:
    data = ClassifyDataUtil(hparams=hparams)
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    num_params = count_params(trainable_params)
    print("Model has {0} params".format(num_params))

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    #optim = torch.optim.Adam(trainable_params, lr=hparams.lr, betas=(0.9, 0.98), eps=1e-9)
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_loss, best_acc, cur_attempt, lr = torch.load(extra_file_name)
  else:
    data = ClassDataUtil(hparams=hparams)
    model = CNNClassify(hparams)
    if args.cuda:
      model = model.cuda()
    #if args.init_type == "uniform":
    #  print("initialize uniform with range {}".format(args.init_range))
    #  for p in model.parameters():
    #    p.data.uniform_(-args.init_range, args.init_range)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    num_params = count_params(trainable_params)
    print("Model has {0} params".format(num_params))

    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    step = 0
    best_loss = None
    best_acc = None
    cur_attempt = 0
    lr = hparams.lr

  #crit = nn.CrossEntropyLoss(reduction='none')
  crit = nn.CrossEntropyLoss(reduce=False)

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  total_loss, total_batch, acc = 0, 0, 0
  model.train()
  epoch = 0
  for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, batch_size, x_train_char_sparse, y_train_char_sparse, eop, file_idx) in data.next_train():
    step += 1
    #print(x_train)
    #print(x_mask)
    logits = model.forward(x_train, x_mask, x_len, x_train_char_sparse, file_idx=file_idx, step=step)
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = Variable(torch.LongTensor(y_train))
    if args.cuda: labels = labels.cuda()
      
    tr_loss = crit(logits, labels)
    _, preds = torch.max(logits, dim=1)
    val_acc = torch.eq(preds, labels).int().sum()

    acc += val_acc.item()
    tr_loss = tr_loss.sum()
    total_loss += tr_loss.item()
    total_batch += batch_size

    tr_loss.div_(batch_size)
    tr_loss.backward()
    grad_norm = grad_clip(trainable_params, grad_bound=args.clip_grad)
    optim.step()
    optim.zero_grad()
    if eop: epoch += 1
    if step % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format((step) / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(total_loss)
      log_string += " acc={0:<5.4f}".format(acc / total_batch)
      log_string += " |g|={0:<5.2f}".format(grad_norm)


      log_string += " wpm(k)={0:<5.2f}".format(total_batch / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
      acc, total_loss, total_batch = 0, 0, 0
      log_start_time = time.time()

    if step % args.eval_every == 0:
      model.eval()
      cur_acc, cur_loss = eval(model, data, crit, step, hparams)
      if not best_acc or best_acc < cur_acc:
        best_loss, best_acc = cur_loss, cur_acc
        cur_attempt = 0
        save_checkpoint([step, best_loss, best_acc, cur_attempt, lr], model, optim, hparams, args.output_dir)
      else:
        if args.lr_dec:
          lr = lr * args.lr_dec
          set_lr(optim, lr)

        cur_attempt += 1
        if args.patience and cur_attempt > args.patience: break
      model.train()

if __name__ == "__main__":
  if not args.decode:
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
  else:
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
    hparams.decode = True
    hparams.test_file_idx_list = [int(i) for i in args.test_file_idx_list.split(",")]
    hparams.test_src_file_list = args.test_src_file_list.split()
    hparams.test_lang_file = args.test_lang_file
    prob_file_list = []
    if hparams.test_lang_file:
      test_src_file_list = []
      with open(hparams.test_lang_file, "r") as myfile:
        for line in myfile:
          lan = line.strip()
          test_src_file_list.append(hparams.test_src_file_list[0].replace("LAN", lan))
          prob_file_list.append(hparams.test_src_file_list[0].replace("LAN", lan) + ".azelogp")
      hparams.test_src_file_list = test_src_file_list

    data = ClassDataUtil(hparams=hparams)
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)

    i = 0
    cur_out = open(prob_file_list[i], "w")
    for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, batch_size, x_char, y_char, eop, eof, dev_file_index in data.next_test(test_batch_size=args.test_batch_size):
      # clear GPU memory
      gc.collect()

      # next batch
      logits = model.forward(
        x, x_mask, x_len, x_char, file_idx=dev_file_index)
      logits = logits.view(-1, hparams.trg_vocab_size)
      prob = nn.functional.log_softmax(logits)
      for p in prob:
        cur_out.write("{}\n".format(p[0].item()))
      if eop: break
      if eof:
        i += 1
        cur_out = open(prob_file_list[i], "w")
