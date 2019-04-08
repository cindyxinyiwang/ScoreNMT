import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import gc
import random
import numpy as np
from utils import *
from model import *
from rl_data_utils import RLDataUtil 
from state_data import *
from featurizer import *
from actor import *
from customAdam import *

from collections import defaultdict, deque
import copy
import time

TEST_STEP = 30
TEST_EVERY = 400
PRINT_EVERY = 200
TRAIN_Q_EPOCH=1
TRAIN_MODEL_EPOCH=2
MAX_ITER = 10
BURN_IN = 500

NEG_R=-2
POS_R=2
EQN_R=0

def get_val_ppl(model, data_loader, hparams, crit, step, load_full, log):
  decode = model.hparams.decode
  model.hparams.decode = True
  model.eval()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0
  total_ppl, total_bleu = 0, 0
  valid_bleu = None
  logits_batch, labels_batch = None, None
  for x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, y_char, eop, eof, dev_file_index, x_rank in data_loader.next_dev(dev_batch_size=hparams.valid_batch_size, load_full=load_full, log=log):
    # clear GPU memory
    #gc.collect()
    # next batch
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    y_count -= batch_size
    # word count
    valid_words += y_count

    logits = model.forward(
      x, x_mask, x_len, x_pos_emb_idxs,
      y[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_char, y_char, file_idx=dev_file_index, step=step, x_rank=x_rank)
    if logits_batch is None:
      logits_batch = logits
      labels_batch = y[:,1:].contiguous()
    logits = logits.view(-1, hparams.trg_vocab_size)
    labels = y[:,1:].contiguous().view(-1)
    val_loss, val_acc = get_performance(crit, logits, labels, hparams)
    n_batches += batch_size
    valid_loss += val_loss.item()
    valid_acc += val_acc.item()
    if eof:
      val_ppl = np.exp(valid_loss / valid_words)
      #print("ppl for dev {}".format(dev_file_index[0]))
      #print("val_step={0:<6d}".format(step))
      #print(" loss={0:<6.2f}".format(valid_loss / valid_words))
      #print(" acc={0:<5.4f}".format(valid_acc / valid_words))
      #print(" val_ppl={0:<.2f}".format(val_ppl))
      if log:
        print(" val_ppl={0:<.2f}".format(val_ppl))
        print(" acc={0:<5.4f}".format(valid_acc / valid_words))
        print(" loss={0:<6.2f}".format(valid_loss / valid_words))
      valid_words = 0
      valid_loss = 0
      valid_acc = 0
      n_batches = 0
      total_ppl = val_ppl       
    if eop:
      break
  model.train()
  model.hparams.decode = decode
  return total_ppl, logits_batch, labels_batch

class ReinforceTrainer():
  def __init__(self, hparams):
    self.hparams = hparams
    self.data_loader = RLDataUtil(hparams)

    if self.hparams.decode:
      print("Decoding...")
      self.nmt_model = torch.load(os.path.join(self.hparams.output_dir, "nmt.pt")) 
      self.nmt_crit = get_criterion(hparams)
      self.actor = torch.load(os.path.join(self.hparams.output_dir, "actor.pt"))
      trainable_params = [
        p for p in self.nmt_model.parameters() if p.requires_grad]
      num_params = count_params(trainable_params)
      print("NMT Model has {0} params".format(num_params))
      self.nmt_optim = torch.optim.Adam(trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)
    else:
      self.nmt_model = Seq2Seq(hparams, self.data_loader)
      print("Training RL...")
      if self.hparams.actor_type == "base":
        self.featurizer = Featurizer(hparams, self.data_loader)
        self.actor = Actor(hparams, self.featurizer.num_feature, self.data_loader.lan_dist_vec)
      elif self.hparams.actor_type == "emb":
        self.featurizer = EmbFeaturizer(hparams, self.nmt_model.encoder.word_emb, self.nmt_model.decoder.word_emb, self.data_loader)
        self.actor = EmbActor(hparams, self.data_loader.lan_dist_vec)
      else:
        print("actor not implemented")
        exit(0)
      if self.hparams.imitate_episode:
        self.heuristic_actor = HeuristicActor(hparams, self.featurizer.num_feature, self.data_loader.lan_dist_vec)
      
      self.start_time = time.time()
      self.init_train_q = True
      trainable_params = [
        p for p in self.nmt_model.parameters() if p.requires_grad]
      num_params = count_params(trainable_params)
      print("NMT Model has {0} params".format(num_params))
      self.nmt_optim = customAdam(trainable_params, hparams, lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)
      
      trainable_params = [
        p for p in self.actor.parameters() if p.requires_grad]
      num_params = count_params(trainable_params)
      print("Actor Model has {0} params".format(num_params))
      self.actor_optim = torch.optim.Adam(trainable_params, lr=self.hparams.lr_q, weight_decay=self.hparams.l2_reg)

      if self.hparams.cuda:
        self.nmt_model = self.nmt_model.cuda()
        self.actor = self.actor.cuda()

      self.cur_attempt = 0
      self.lr = self.hparams.lr
      self.best_val_ppl = None
      if hparams.init_type == "uniform" and not hparams.model_type == "transformer":
        print("initialize uniform with range {}".format(hparams.init_range))
        for p in self.nmt_model.parameters():
          p.data.uniform_(-hparams.init_range, hparams.init_range)
      self.init_train_lr()
    self.best_val_ppl = [None for _ in range(len(hparams.dev_src_file_list))]
    self.best_val_bleu = [None for _ in range(len(hparams.dev_src_file_list))]

  def init_train_lr(self):
    self.cur_temp = self.hparams.min_temp
    self.cur_step = 0
   
  def decode(self, output_file):
    max_step = 15
    train_step = 0
    step = 0
    output = output_file
    output = open(output, 'w')
    data_batch, next_data_batch = [], []
    batch_count = 0
    hparams = copy.deepcopy(self.hparams)
    hparams.decode = True
    data_util = RLDataUtil(hparams)
    #for src, src_len, trg, iter_percent, eop in self.data_loader.next_raw_example():
    for src, src_len, trg, iter_percent, eop in data_util.next_raw_example():
      #print(src, src_len, trg, iter_percent, eop)
      step += 1
      s = self.featurizer.get_state(src, src_len, trg)
      a_logits = self.actor(s)
      mask = 1 - s[1].byte()
      a_logits.masked_fill_(mask, -float("inf"))

      a, prob = sample_action(a_logits, temp=self.hparams.max_temp, log=False)
      batch_size = len(src)
      prob_f = [repr(i) for i in prob]

      for _ in range(batch_size):
        output.write(" ".join(prob_f) + "\n")

      if eop: break  
   
  def train_score(self):
    step = 0
    # update the actor with graidents scaled by cosine similarity
    # first update on the base language
    for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id, eop) in self.data_loader.next_base_data():
      logits = self.nmt_model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, [], [], file_idx=[], step=step, x_rank=[])
      logits = logits.view(-1, self.hparams.trg_vocab_size)
      labels = y_train[:,1:].contiguous().view(-1)
      cur_nmt_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=self.hparams.pad_id, reduction="none")
      cur_nmt_loss = cur_nmt_loss.view(batch_size, -1).sum().div_(batch_size * self.hparams.update_batch)
      # save the gradients to nmt moving average
      cur_nmt_loss.backward()
      grad_norm = torch.nn.utils.clip_grad_norm_(self.nmt_model.parameters(), self.hparams.clip_grad)
      self.nmt_optim.save_gradients(self.hparams.base_lan_id)
      break

    grad_cosine_sim = self.nmt_optim.get_cosine_sim()
    grad_scale = torch.stack([grad_cosine_sim[idx] for idx in range(self.hparams.lan_size)]).view(1, -1)
    print(grad_scale.data)
    for src, src_len, trg, iter_percent, eop in self.data_loader.next_raw_example():
      s = self.featurizer.get_state(src, src_len, trg)
      step += 1
      a_logits = self.actor(s)
      mask = 1 - s[1].byte()
      a_logits.masked_fill_(mask, -float("inf"))
      
      loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
      loss = (loss * grad_scale * 0.01).masked_fill_(mask, 0.).sum() 
      cur_loss = loss.item()
      loss.backward()
      self.actor_optim.step()
      self.actor_optim.zero_grad()
      if step % self.hparams.print_every == 0:
        print("actor loss={}".format(cur_loss))
      if eop: break
  
  def imitate_heuristic(self):
    data_batch, next_data_batch = [], []
    batch_count = 0
    step = 0
    cur_dev_ppl = 12
    for eps in range(self.hparams.imitate_episode):
      for src, src_len, trg, iter_percent, eop in self.data_loader.next_raw_example():
        s = self.featurizer.get_state(src, src_len, trg)
        step += 1
        #if step % self.hparams.print_every == 0:
        #  print(s[1])
        a_logits = self.actor(s)
        a_target_logits = self.heuristic_actor(s)
        mask = 1 - s[1].byte()
        a_logits.masked_fill_(mask, -float("inf"))
        a_prob = torch.nn.functional.softmax(a_logits, dim=-1)
        a_target_logits.masked_fill_(mask, -float("inf"))
        a_target_prob = torch.nn.functional.softmax(a_target_logits, dim=-1)
        loss = torch.nn.functional.mse_loss(a_prob, a_target_prob)
        cur_loss = loss.item()
        loss.backward()
        self.actor_optim.step()
        self.actor_optim.zero_grad()
        #if step % self.hparams.print_every == 0:
        #  print("step={} imitation loss={}".format(step, cur_loss))

        if eop: break  
    return

  def init_train_nmt(self):
    self.step = 0
    self.cur_attempt = 0
    self.lr = self.hparams.lr
    self.epoch = 0

  def train_nmt_full(self, output_prob_file, n_train_epochs):
    hparams = copy.deepcopy(self.hparams)

    hparams.train_nmt = True
    hparams.output_prob_file = output_prob_file
    hparams.n_train_epochs = n_train_epochs

    model = self.nmt_model 
    optim = self.nmt_optim
    #optim = torch.optim.Adam(trainable_params)
    #step = 0
    #cur_attempt = 0
    #lr = hparams.lr
  
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
    #epoch = 0
    update_batch_size = 0
    #for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id, eop) in data_util.next_nmt_train():
    for (x_train, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, lan_id, eop, eob, save_grad) in self.data_loader.next_sample_nmt_train(self.featurizer, self.actor):
      self.step += 1
      target_words += (y_count - batch_size)
      logits = model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, [], [], file_idx=[], step=self.step, x_rank=[])
      logits = logits.view(-1, hparams.trg_vocab_size)
      labels = y_train[:,1:].contiguous().view(-1)
 
      cur_nmt_loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=self.hparams.pad_id, reduction="none")
      total_loss += cur_nmt_loss.sum().item()
      cur_nmt_loss = cur_nmt_loss.view(batch_size, -1).sum(-1).div_(batch_size * hparams.update_batch)
     
      if save_grad:
        # save the gradients to nmt moving average
        for batch_id in range(batch_size):
          batch_lan_id = lan_id[batch_id]
          cur_nmt_loss[batch_id].backward(retain_graph=True)
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip_grad)
          optim.save_gradients(batch_lan_id)
      else:
        cur_nmt_loss = cur_nmt_loss.sum()
        cur_nmt_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip_grad)

      mask = (labels == hparams.pad_id)
      _, preds = torch.max(logits, dim=1)
      cur_tr_acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()

      total_corrects += cur_tr_acc.item()

      if self.step % hparams.update_batch == 0:
        optim.step()
        optim.zero_grad()
        update_batch_size = 0
      # clean up GPU memory
      if self.step % hparams.clean_mem_every == 0:
        gc.collect()
      if eop: 
        self.epoch += 1
      #  get_grad_cos_all(model, data, crit)
      if (self.step / hparams.update_batch) % hparams.log_every == 0:
        curr_time = time.time()
        since_start = (curr_time - start_time) / 60.0
        elapsed = (curr_time - log_start_time) / 60.0
        log_string = "ep={0:<3d}".format(self.epoch)
        log_string += " steps={0:<6.2f}".format((self.step / hparams.update_batch) / 1000)
        log_string += " lr={0:<9.7f}".format(self.lr)
        log_string += " loss={0:<7.2f}".format(cur_nmt_loss.sum().item())
        log_string += " |g|={0:<5.2f}".format(grad_norm)
  
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_words)
  
        log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
        log_string += " time(min)={0:<5.2f}".format(since_start)
        print(log_string)
      if hparams.eval_end_epoch:
        if eop:
          eval_now = True
        else:
          eval_now = False
      elif (self.step / hparams.update_batch) % hparams.eval_every == 0:
        eval_now = True
      else:
        eval_now = False 
      if eval_now:
        based_on_bleu = hparams.eval_bleu and self.best_val_ppl[0] is not None and self.best_val_ppl[0] <= hparams.ppl_thresh
        with torch.no_grad():
          val_ppl, val_bleu, ppl_list, bleu_list = eval(model, data_util, self.step, hparams, hparams, eval_bleu=based_on_bleu, valid_batch_size=hparams.valid_batch_size, tr_logits=logits)	
        for i in range(len(ppl_list)):
          if based_on_bleu:
            if self.best_val_bleu[i] is None or self.best_val_bleu[i] <= bleu_list[i]:
              save = True 
              self.best_val_bleu[i] = bleu_list[i]
              self.cur_attempt = 0
            else:
              save = False
              self.cur_attempt += 1
          else:
            if self.best_val_ppl[i] is None or self.best_val_ppl[i] >= ppl_list[i]:
              save = True
              self.best_val_ppl[i] = ppl_list[i]
              self.cur_attempt = 0 
            else:
              save = False
              self.cur_attempt += 1
          if save:
            if len(ppl_list) > 1:
              nmt_save_checkpoint([self.step, self.best_val_ppl, self.best_val_bleu, self.cur_attempt, self.lr], model, optim, hparams, hparams.output_dir + "dev{}".format(i))
            else:
              nmt_save_checkpoint([self.step, self.best_val_ppl, self.best_val_bleu, self.cur_attempt, self.lr], model, optim, hparams, hparams.output_dir)
          elif not hparams.lr_schedule and self.step >= hparams.n_warm_ups:
            self.lr = self.lr * hparams.lr_dec
            set_lr(optim, self.lr)
        # reset counter after eval
        log_start_time = time.time()
        target_words = total_corrects = total_loss = 0
        target_rules = target_total = target_eos = 0
        total_word_loss = total_rule_loss = total_eos_loss = 0
      if hparams.patience >= 0:
        if self.cur_attempt > hparams.patience: break
      elif hparams.n_train_epochs > 0:
        if self.epoch >= hparams.n_train_epochs: break
      else:
        if self.step > hparams.n_train_steps: break
      if eob: break

  def train_rl_and_nmt(self):
    self.init_train_nmt()
    # imitate a good policy agent first
    if self.hparams.imitate_episode:
      self.imitate_heuristic()
    while True:
      # use current policy to load data
      #output_prob_file = self.hparams.output_prob_file + str(iteration)
      #self.decode(output_prob_file)
      # update nmt with current data, and update the policy
      self.train_nmt_full(self.hparams.output_prob_file, n_train_epochs=(self.hparams.n_train_epochs // self.hparams.iteration))
      if self.hparams.patience >= 0:
        if self.cur_attempt > self.hparams.patience: break
      elif self.hparams.n_train_epochs > 0:
        if self.epoch >= self.hparams.n_train_epochs: break
      else:
        if self.step > self.hparams.n_train_steps: break
      #self.imitate_episode = 2
      #self.imitate_heuristic()
      self.train_score()

