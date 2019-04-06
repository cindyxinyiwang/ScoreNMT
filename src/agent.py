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
from mult_data_utils import MultDataUtil 
from state_data import *

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
  return total_ppl, logits_batch, labels_batch

class Featurizer(nn.Module):
  def __init__(self, hparams, total_train_count, data_loader):
    super(Featurizer, self).__init__()
    self.hparams = hparams
    self.state_data = StateData(hparams, data_loader)
    self.total_train_count = total_train_count

    self.reset_feature()
    #self.num_feature = self.state_data.num_feature + (self.hparams.valid_batch_size + 2) 
    if self.hparams.add_dev_logit_feature:
      self.num_feature = self.state_data.num_feature + 2 + self.hparams.valid_batch_size 
    else:
      self.num_feature = self.state_data.num_feature + 2 
    #if self.hparams.add_model_feature:
    #  self.marginal_conv_list = []
    #  for k in [3]:
    #    self.marginal_conv_list.append(torch.nn.Conv2d(1, out_channels=20, kernel_size=(k, self.hparams.trg_vocab_size)))
    #  self.marginal_conv_list = nn.ModuleList(self.marginal_conv_list)

    #  self.prob_conv_list = []
    #  for k in [4, 5]:
    #    self.prob_conv_list.append(torch.nn.Conv1d(1, out_channels=20, kernel_size=k))
    #  self.prob_conv_list = nn.ModuleList(self.prob_conv_list)
 
  def get_model_features_rnn(self, dev_logits, dev_labels):
    # dev_logits: (batch_size, len, vocab_size)
    # dev_labels: (batch_size, len)
    dev_logits = dev_logits.data
    dev_labels = dev_labels.data
    batch_size = dev_logits.size(0)
    # marginals
    marginal_outputs = []
    for conv in self.marginal_conv_list:
      # (batch_size, c_out, len-k, 1)
      conved = conv(dev_logits.unsqueeze(1)).squeeze(3)
      # (batch_size, c_out, 1)
      pooled = nn.functional.avg_pool1d(conved, kernel_size=conved.size(2)).squeeze(2)
      # (c_out,)
      marginal_outputs.append(pooled.sum(0, keepdim=True) / batch_size)
    # max probs
    logits = dev_logits.view(-1)
    labels = dev_labels.view(-1)
    offset = torch.arange(labels.size(0)) * self.hparams.trg_vocab_size
    if self.hparams.cuda: offset = offset.cuda()
    labels = labels + offset
    target_label_logits = logits[labels].view(batch_size, -1)

    #ave_label_logits = label_logits.sum(1) / label_logits.size(1)
    #label_logits = label_logits - ave_label_logits
    prob_outputs = []
    for conv in self.prob_conv_list:
      conved = conv(target_label_logits.unsqueeze(1))
      # (batch_size, c_out, 1)
      pooled = nn.functional.max_pool1d(conved, kernel_size=conved.size(2)).squeeze(2)
      # (c_out, )
      prob_outputs.append(pooled.sum(0, keepdim=True) / batch_size)
    return torch.cat(marginal_outputs + prob_outputs, dim=1)
 
  def get_model_features(self, dev_logits, dev_labels):
    # dev_logits: (batch_size, len, vocab_size)
    # dev_labels: (batch_size, len)
    batch_size = dev_logits.size(0)
    dev_std = dev_logits.data.std(dim=-1)
    dev_std_mean = dev_std.mean(dim=-1)
    dev_feature = dev_std_mean.unsqueeze(0)
    
    features = []
    train_percent = self.count / self.total_train_count
    features.append(train_percent)
    update_percent = self.updated_count / self.total_train_count
    features.append(update_percent)
    features = torch.FloatTensor([features])
    if self.hparams.cuda: features = features.cuda()

    if self.hparams.add_dev_logit_feature:
      ret = torch.cat([dev_feature, features], dim=1)
    else:
      ret = features
    return ret
    #ret = torch.FloatTensor([[0]])
    #if self.hparams.cuda: ret = ret.cuda()
    #return ret 

  def get_language_features(self, lan_id, data_loader):
    # language level distance with base
    base_lan_id = data_loader.lan_w2i[self.hparams.base_lan]
    lan_dist = data_loader.query_lan_dist(base_lan_id, lan_id)
    return [[lan_dist]]

  def get_data_features(self, x_train_list, x_char, y_train_list, eop, lan_id, update_ngram=True):
    # diversity for y
    #sent_appeared_num, batch_size = 0, 0
    #for y in y_train_list:
    #  y = hash(tuple(y))
    #  sent_appeared_num += self.y_dict[y] / self.step
    #  self.y_dict[y] += 1
    #  batch_size += 1
    ## diversity for y ngram
    #weighted_ngram_appeared_num, ngram_appeared_num, ngram_count = 0, 0, 0
    #for y in y_train_list:
    #  for n in range(4):
    #    for i in range(len(y)-n):
    #      ngram = hash(tuple(y[i:i+n]))
    #      ngram_appeared_num += self.y_ngram_dict[ngram] / self.step
    #      weighted_ngram_appeared_num += self.y_weighted_ngram_dict[ngram] / self.step
    #      ngram_count += 1
    #      
    #      self.y_ngram_dict[ngram] += 1
    #      self.y_weighted_ngram_dict[ngram] += lan_dist/100

    features = self.state_data.get_target_features(x_train_list, y_train_list, eop, update_ngram=update_ngram)
    features_src = self.state_data.get_source_features(x_train_list, y_train_list, eop, update_ngram=update_ngram)
    features_src_static = self.state_data.get_source_static_features(x_train_list, y_train_list, eop, lan_id, update_ngram=update_ngram)
    features.extend(features_src)
    features.extend(features_src_static)
    #features.extend([sent_appeared_num / batch_size, ngram_appeared_num / ngram_count, weighted_ngram_appeared_num / ngram_count])
    return [features]

  def forward(self, model, data_loader, lan, x_train_list, x_char, y_train_list, eop, dev_logits=None, dev_labels=None, update_ngram=True):
  #def forward(self, model, data_loader, lan, x_train, x_mask, x_len, y_train, y_mask, y_len, eop):
    if eop: self.reset_feature()
    data_feature = self.state_data.get_data_features(x_train_list, x_char, y_train_list, eop, lan, update_ngram=update_ngram)

    if self.hparams.add_model_feature:
      model_feature = self.get_model_features(dev_logits, dev_labels)
      #model_feature = Variable(model_feature)
      #model_feature = Variable(torch.FloatTensor(model_feature))
    data_feature = Variable(torch.FloatTensor(data_feature))
    if self.hparams.cuda: 
      if self.hparams.add_model_feature:
        model_feature = model_feature.cuda()
      data_feature = data_feature.cuda()
    if self.hparams.add_model_feature:
      return torch.cat([model_feature, data_feature], dim=1)
    else:
      return data_feature

  def update_feature_state(self, action, x_train_list):
    batch_size = len(x_train_list)
    if action == 1:
      self.updated_count += batch_size
    self.count += batch_size

  def reset_feature(self):
    self.y_dict = defaultdict(int)
    self.y_ngram_dict = defaultdict(int)
    self.y_weighted_ngram_dict = defaultdict(int)
    self.state_data.reset_feature()
    self.count = 0
    self.updated_count = 0

class Reward():
  def __init__(self, hparams):
    self.hparams = hparams
    self.max_len = 20
    self.past_rewards = deque([10])

  def init_reward(self):
    pass

  def update_reward(self):
    pass

  def get_reward(self, val_ppl, new_val_ppl):
    if self.hparams.reward_type == "fixed":
      if new_val_ppl > val_ppl:
        r = self.hparams.neg_r
      elif new_val_ppl == val_ppl:
        r = self.hparams.eqn_r
      else:
        r = self.hparams.pos_r
    elif self.hparams.reward_type == "dynamic":
      if new_val_ppl != val_ppl:
        self.past_rewards.append(abs(new_val_ppl - val_ppl))
        if len(self.past_rewards) > self.max_len:
          self.past_rewards.popleft()
        ave = np.mean(self.past_rewards)
        if val_ppl > new_val_ppl:
          r = abs((val_ppl - new_val_ppl) - ave)
        else:
          r = -abs((new_val_ppl - val_ppl) - ave)
        print(r, ave)
      else:
        r = 0
        print(r)
    return r

class QNN(nn.Module):
  def __init__(self, hparams, num_feature):
    super(QNN, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden
    self.w = nn.Linear(num_feature, hidden_size)
    self.w2 = nn.Linear(hidden_size, hidden_size)
    self.decoder = nn.Linear(hidden_size, 2)
    # init
    for p in self.decoder.parameters():
      init.uniform_(p, -0.05, 0.05)
    for p in self.w.parameters():
      init.uniform_(p, -0.05, 0.05)
    for p in self.w2.parameters():
      init.uniform_(p, -0.05, 0.05)
    #self.decoder.bias = torch.nn.Parameter(torch.FloatTensor([0, 0.1]))

  def forward(self, feature):
    #(model_feature, language_feature, data_feature) = feature
    enc = self.w(feature)
    enc = torch.relu(enc)
    enc = self.w2(enc)
    enc = torch.relu(enc)
    logit = self.decoder(enc)
    return logit

class AgentTrainer():
  def __init__(self, hparams):
    self.hparams = hparams
    self.data_loader = MultDataUtil(hparams)
    self.model_data_loader = MultDataUtil(hparams)
    self.model = Seq2Seq(hparams, self.data_loader)
    self.reward = Reward(hparams)
    if (not self.hparams.train_file_idx):
      if self.hparams.agent_subsample_percent:
        self.featurizer = Featurizer(hparams, self.hparams.agent_subsample_percent * self.data_loader.total_train_count, self.data_loader)
      else:
        self.featurizer = Featurizer(hparams, self.hparams.agent_subsample_line * self.hparams.lan_size, self.data_loader)
      self.model_featurizer = Featurizer(hparams,self.model_data_loader.total_train_count, self.model_data_loader)
      self.Q = QNN(hparams, self.featurizer.num_feature)
      self.Q_target = QNN(hparams, self.featurizer.num_feature) 
    self.start_time = time.time()
    self.init_train_q = True

    if hparams.init_type == "uniform" and not hparams.model_type == "transformer":
      print("initialize uniform with range {}".format(hparams.init_range))
      for p in self.model.parameters():
        p.data.uniform_(-hparams.init_range, hparams.init_range)

  def init_policy(self, cur_data):
    # pretrain Q policy with the best heuristic policy
    # given the data, decide the action
    pass

  def burn_in(self, crit):
    step = 0
    val_ppl = float("inf")
    # keep a copy of current NMT state
    orig_model = copy.deepcopy(self.model.state_dict())
    prev_data = None
    for cur_data in self.data_loader.next_train(load_full=False):
      #if prev_data is None: 
      #  prev_data = cur_data
      #  continue
      (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) = cur_data
      step += 1
      if step < 20:
        a = 1
        cont = True
      else:
        cont = False
        val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, crit, step, load_full=False, log=False)
        s_i = self.featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=True)
        a = self.epsilon_greedy(s_i, self.Q, self.hparams.epsilon_max)
        self.featurizer.update_feature_state(a, x_train_list)
      if a == 1:
        # update the model
        target_words = (y_count - batch_size)
        logits = self.model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
        logits = logits.view(-1, self.hparams.trg_vocab_size)
        labels = y_train[:,1:].contiguous().view(-1)
        cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, self.hparams)
        cur_tr_loss.div_(batch_size)
        cur_tr_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad)
        self.optim_model.step()
        self.optim_model.zero_grad()
        if cont: continue
      new_val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, crit, step, load_full=False, log=False)
      r = self.reward.get_reward(val_ppl, new_val_ppl)
      (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train_train, y_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) = cur_data
      s_i1 = self.featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=False)
      if self.hparams.balance_sample:
        if a == 0:
          self.replay_mem_0.append((s_i, a, r, s_i1))
        else:
          self.replay_mem_1.append((s_i, a, r, s_i1))
        cur_size = len(self.replay_mem_0) + len(self.replay_mem_1)
      else:
        self.replay_mem.append((s_i, a, r, s_i1))
        cur_size = len(self.replay_mem)
      if cur_size % self.hparams.print_every == 0:
        print("burn in={}".format(cur_size))
      prev_data = cur_data
      if cur_size >= self.hparams.burn_in_size: break
    # reset episode
    #for m_p, o_p in zip(self.model.parameters(), orig_model.parameters()):
    #  m_p.data = o_p.data
    if self.hparams.reset_nmt:
      self.model.load_state_dict(orig_model)

  def test(self, optim, crit):
    val_ppl = float("inf")
    total_r = 0
    orig_model = copy.deepcopy(self.model.state_dict())
    step = 0
    prev_data = None
    for (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) in self.data_loader.next_train():
      step += 1
      val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, crit, step, load_full=False, log=False)
      if self.hparams.add_model_feature:
        pass
      else:
        dev_logits, dev_labels = None, None

      s_i = self.featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=True)
      a = self.greedy(s_i, self.Q)

      if a == 1:
        # update the model
        target_words = (y_count - batch_size)
        logits = self.model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
        logits = logits.view(-1, self.hparams.trg_vocab_size)
        labels = y_train[:,1:].contiguous().view(-1)
        cur_tr_loss, cur_tr_acc = get_performance(crit, logits, labels, self.hparams)
        cur_tr_loss.div_(batch_size)
        cur_tr_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad)
        self.optim_model.step()
        self.optim_model.zero_grad()

      new_val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, crit, step, load_full=False, log=False)
      if new_val_ppl > val_ppl:
        r = self.hparams.neg_r
      elif new_val_ppl == val_ppl:
        r = self.hparams.eqn_r
      else:
        r = self.hparams.pos_r
      val_ppl = new_val_ppl
      total_r += r
      if step == self.hparams.test_step: break
    #for m_p, o_p in zip(self.model.parameters(), orig_model.parameters()):
    #  m_p.data = o_p.data
    if self.hparams.reset_nmt:
      self.model.load_state_dict(orig_model)

    return total_r

  def init_train_game(self):
    trainable_params = [
      p for p in self.model.parameters() if p.requires_grad]
    num_params = count_params(trainable_params)
    print("Model has {0} params".format(num_params))
    self.optim_model = torch.optim.Adam(trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.l2_reg)
    if (not self.hparams.train_file_idx):
      trainable_params = [
        p for p in self.Q.parameters() if p.requires_grad]
      self.optim_q = torch.optim.Adam(trainable_params, lr=self.hparams.lr_q, weight_decay=self.hparams.l2_reg)
      if self.hparams.double_q:
        trainable_params = [
          p for p in self.Q_target.parameters() if p.requires_grad]
        self.optim_q_target = torch.optim.Adam(trainable_params, lr=self.hparams.lr_q, weight_decay=self.hparams.l2_reg)
    self.crit = get_criterion(self.hparams)
    self.q_crit = get_criterion_q(self.hparams)
    if (not self.hparams.train_file_idx):
      self.Q.train()
      self.Q_target.train()
    if self.hparams.cuda:
      self.model = self.model.cuda()
      if (not self.hparams.train_file_idx):
        self.Q = self.Q.cuda()
        self.Q_target = self.Q_target.cuda()
    if self.hparams.balance_sample:
      self.replay_mem_0 = deque([])
      self.replay_mem_1 = deque([])
    else:
      self.replay_mem = deque([])
    self.cur_attempt = 0
    self.lr = self.hparams.lr
    self.best_val_ppl = None
    self.train_q_step = 0
    self.updated_step = 0
    self.record_action = True

  def prepare_train_q(self):
    self.train_q_step = 0

  def greedy(self, s, Q):
    a_logit = Q.forward(s).view(-1).data
    if a_logit[0] < a_logit[1]:
      a = 1
    else:
      a = 0
    return a

  def epsilon_greedy(self, s, Q, epsilon):
    a_logit = Q.forward(s).view(-1).data
    if random.random() < epsilon:
      a = random.randint(0, a_logit.size(0)-1)
    else:
      if a_logit[0] < a_logit[1]:
        a = 1
      else:
        a = 0
    return a

  def train_q(self):
    val_ppl = float("inf")
    # sync Q param and Q_target param
    self.Q_target.load_state_dict(self.Q.state_dict())
    self.featurizer.reset_feature()
    # start training q network
    prev_data = None
    # keep a copy of current NMT state
    orig_model = copy.deepcopy(self.model.state_dict())
    epoch = 0
    for cur_data in self.data_loader.next_train(load_full=False, log=True):
      #if prev_data is None: 
      #  prev_data = cur_data
      #  continue
      #if self.hparams.double_q:
      #  if random.random() < 0.5:
      #    t = self.Q
      #    self.Q = self.Q_target
      #    self.Q_target = t
      #    t = self.optim_q
      #    self.optim_q = self.optim_q_target
      #    self.optim_q_target = t
      (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) = cur_data
      self.train_q_step += 1
      #if self.train_q_step % self.hparams.clean_mem_every == 0:
      #  gc.collect()

      val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, self.crit, self.train_q_step, load_full=False, log=False)
      s_i = self.featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=True)
      
      step_ratio = max(0, (self.hparams.epsilon_anneal_step - self.train_q_step))
      epsilon = self.hparams.epsilon_min + step_ratio / self.hparams.epsilon_anneal_step * (self.hparams.epsilon_max - self.hparams.epsilon_min)
      a = self.epsilon_greedy(s_i, self.Q, epsilon)

      self.featurizer.update_feature_state(a, x_train_list)
      
      if a == 1:
        # update the model
        target_words = (y_count - batch_size)
        logits = self.model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=self.train_q_step, x_rank=x_rank)
        logits = logits.view(-1, self.hparams.trg_vocab_size)
        labels = y_train[:,1:].contiguous().view(-1)
        cur_tr_loss, cur_tr_acc = get_performance(self.crit, logits, labels, self.hparams)
        cur_tr_loss.div_(batch_size)
        cur_tr_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad)
        self.optim_model.step()
        self.optim_model.zero_grad()

      new_val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, self.crit, self.train_q_step, load_full=False, log=False)
      if self.train_q_step % self.hparams.print_every == 0:
        print("val ppl={}".format(new_val_ppl))

      r = self.reward.get_reward(val_ppl, new_val_ppl)
      (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) = cur_data
      s_i1 = self.featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=False)
      if self.hparams.balance_sample:
        if a == 0:
          self.replay_mem_0.append((s_i, a, r, s_i1))
        else:
          self.replay_mem_1.append((s_i, a, r, s_i1))
        if len(self.replay_mem_0) + len(self.replay_mem_1) > self.hparams.replay_mem_size:
          if len(self.replay_mem_0) > len(self.replay_mem_1):
            self.replay_mem_0.popleft()
          else:
            self.replay_mem_1.popleft()
        replay_batch_size = int(self.hparams.replay_batch_size / 2)
        if len(self.replay_mem_0) < replay_batch_size:
          mem_list_0 = self.replay_mem_0
        else:
          mem_list_0 = random.sample(self.replay_mem_0, replay_batch_size)
        if len(self.replay_mem_1) < replay_batch_size:
          mem_list_1 = self.replay_mem_1
        else:
          mem_list_1 = random.sample(self.replay_mem_1, replay_batch_size)
        mem_list = list(mem_list_0) + list(mem_list_1)
        random.shuffle(mem_list)
      else:
        self.replay_mem.append((s_i, a, r, s_i1))
        if len(self.replay_mem) > self.hparams.replay_mem_size: 
          self.replay_mem.popleft()
        mem_list = random.sample(self.replay_mem, self.hparams.replay_batch_size)

      mem_si, mem_a, mem_r, mem_si1 = [], [], [], []
      for m in mem_list:
        mem_si.append(m[0])
        mem_a.append(m[1])
        mem_r.append(m[2])
        mem_si1.append(m[3])
      mem_si = torch.cat(mem_si, dim=0)
      mem_si1 = torch.cat(mem_si1, dim=0)
      mem_r = torch.FloatTensor(mem_r)
      mem_a = torch.LongTensor(mem_a)
      if self.hparams.cuda:
        mem_r = mem_r.cuda()
        mem_a = mem_a.cuda()
      replay_batch_size = mem_si.size(0)

      q_vals = self.Q.forward(mem_si)
      #offset = torch.arange(mem_si.size(0)) * q_vals.size(1)
      #if self.hparams.cuda: offset = offset.cuda()
      #mem_a_idx = mem_a.view(-1) + offset
      #q_val = q_vals.view(-1)[mem_a_idx].view(mem_si.size(0), 1)
      
      q_targets = q_vals.data.clone()
      if self.hparams.double_q:
        _, max_idx = torch.max(self.Q.forward(mem_si1), dim=1, keepdim=True)
        q_vs = self.Q_target.forward(mem_si1)
        q_future = torch.cat([torch.index_select(a.unsqueeze(0), 1, i).unsqueeze(0) for a, i in zip(q_vs, max_idx)])
      else:
        q_future, _ = torch.max(self.Q_target.forward(mem_si1), dim=1, keepdim=True)
      
      for i in range(replay_batch_size):
        q_targets[i][mem_a[i].item()] = mem_r[i] + self.hparams.gamma * q_future[i]
      
      q_loss = get_performance_q(self.q_crit, q_targets, q_vals, self.hparams)
      q_loss.div_(self.hparams.replay_batch_size)
      if self.train_q_step % self.hparams.print_every == 0:
        print("q_loss: {}".format(q_loss.item()))
        print("q_vals", q_vals.data)
        print("a", mem_a)
        print("r", mem_r)
      q_loss.backward()
      self.optim_q.step()
      self.optim_q.zero_grad()

      # sync Q param and Q_target param
      if self.train_q_step % self.hparams.sync_target_every == 0:
        self.Q_target.load_state_dict(self.Q.state_dict())

      #if self.train_q_step % self.hparams.test_every == 0:
      #  reward = self.test(self.optim_model, self.crit)
      #  print("test reward={}".format(reward))
      prev_data = cur_data
      if eop:
        epoch += 1
        # reset episode
        #for m_p, o_p in zip(self.model.parameters(), orig_model.parameters()):
        #  m_p.data = o_p.data
        if self.hparams.reset_nmt:
          self.model.load_state_dict(orig_model)

        if epoch >= self.hparams.train_q_epoch: 
          if self.init_train_q:
            self.init_train_q = False
          else:
            break

  def train_model(self, iteration, train_epoch=0):
    step = self.updated_step
    updated_step = 0
    val_ppl = float("inf")
    # start training q network
    prev_data = None
    epoch = 0
    total_loss, total_corrects, target_words, log_start_time = 0, 0, 0, time.time()
    self.model_data_loader.record_action = self.record_action
    for cur_data in self.model_data_loader.next_train(load_full=True):
      #if prev_data is None: 
      #  prev_data = cur_data
      #  continue
      #(x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, action) = cur_data
      (x_train, x_train_list, x_mask, x_count, x_len, x_pos_emb_idxs, y_train, y_train_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_train_char_sparse, y_train_char_sparse, eop, eof, file_idx, x_rank, data_idx, batch_idx, action) = cur_data

      if self.hparams.train_file_idx:
        if self.hparams.train_file_idx[0] == -1 or file_idx[0] in self.hparams.train_file_idx:
          a = 1
          step += 1
        else:
          a = 0
      else:
        if self.record_action:
          val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, self.crit, step, load_full=False, log=False)
          s_i = self.model_featurizer(self.model, self.data_loader, file_idx[0], x_train_list, x_train_char_sparse, y_train_list, eop, dev_logits, dev_labels, update_ngram=True)
          a = self.greedy(s_i, self.Q)
          self.model_data_loader.update_action(data_idx, batch_idx, a)
        else:
          a = action
        self.model_featurizer.update_feature_state(a, x_train_list)
        step += 1

      if a == 1:
        updated_step += 1
        self.updated_step += 1
        # update the model
        target_words += (y_count - batch_size)
        logits = self.model.forward(x_train, x_mask, x_len, x_pos_emb_idxs, y_train[:,:-1], y_mask[:,:-1], y_len, y_pos_emb_idxs, x_train_char_sparse, y_train_char_sparse, file_idx=file_idx, step=step, x_rank=x_rank)
        logits = logits.view(-1, self.hparams.trg_vocab_size)
        labels = y_train[:,1:].contiguous().view(-1)
        cur_tr_loss, cur_tr_acc = get_performance(self.crit, logits, labels, self.hparams)
        total_loss += cur_tr_loss.item()
        total_corrects += cur_tr_acc.item()
        cur_tr_loss.div_(batch_size)

        cur_tr_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad)
        self.optim_model.step()
        self.optim_model.zero_grad()

      if (not self.hparams.train_file_idx) and self.hparams.train_q_every and step % self.hparams.train_q_every == 0:
        self.prepare_train_q()
        self.train_q()

      if (step) % self.hparams.log_every == 0 and target_words > 0:
        curr_time = time.time()
        since_start = (curr_time - self.start_time) / 60.0
        elapsed = (curr_time - log_start_time) / 60.0
        log_string = "ep={0:<3d}".format(epoch+self.hparams.train_model_epoch*iteration)
        log_string += " steps={0:<6.2f}".format((step) / 1000)
        log_string += " update_step={}".format((self.updated_step) / 1000)
        log_string += " lr={0:<9.7f}".format(self.lr)
        log_string += " loss={0:<7.2f}".format(cur_tr_loss.item())
        log_string += " |g|={0:<5.2f}".format(grad_norm)
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_words)

        log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
        log_string += " time(min)={0:<5.2f}".format(since_start)
        print(log_string)
      elif (step) % self.hparams.log_every == 0 and target_words == 0: 
        curr_time = time.time()
        since_start = (curr_time - self.start_time) / 60.0
        elapsed = (curr_time - log_start_time) / 60.0
        log_string = "ep={0:<3d}".format(epoch+self.hparams.train_model_epoch*iteration)
        log_string += " steps={0:<6.2f}".format((step) / 1000)
        log_string += " update_step={}".format((self.updated_step))
        print(log_string)

      if self.updated_step % self.hparams.eval_every == 0 and target_words > 0:
        new_val_ppl, dev_logits, dev_labels = get_val_ppl(self.model, self.data_loader, self.hparams, self.crit, step, load_full=True, log=True)
        print("val step={}".format(step))
        print("updated step={}".format(self.updated_step))
        print("val ppl={}".format(new_val_ppl))
        if self.best_val_ppl is None or self.best_val_ppl >= new_val_ppl:
          save = True
          self.best_val_ppl = new_val_ppl
          self.cur_attempt = 0 
        else:
          save = False
          self.cur_attempt += 1
        if save:
          save_checkpoint(self.model, self, self.hparams, self.hparams.output_dir, save_agent=False)
        else:
          self.lr = self.lr * self.hparams.lr_dec
          set_lr(self.optim_model, self.lr)
        total_loss, total_corrects, target_words, log_start_time = 0, 0, 0, time.time()
        self.updated_step = 0
        if self.hparams.patience>=0 and self.cur_attempt > self.hparams.patience: return 
      if eop:
        epoch += 1
        self.record_action = False
        self.model_data_loader.record_action = self.record_action
        if epoch >= self.hparams.train_model_epoch or (train_epoch and epoch >= train_epoch): break
      prev_data = cur_data

  def train(self):
    self.init_train_game()
    iteration = 0
    if self.hparams.pretrain_nmt_epoch:
      orig_idx = self.hparams.train_file_idx
      self.hparams.train_file_idx = [-1]
      self.train_model(iteration=0, train_epoch=self.hparams.pretrain_nmt_epoch)
      self.hparams.train_file_idx = orig_idx
    # burn in for replay mem
    if (not self.hparams.train_file_idx):
      self.burn_in(self.crit)
    while True:
      iteration += 1
      if (not self.hparams.train_file_idx):
        self.prepare_train_q()
        print("training Q...")
        self.train_q()
      print("training Model...")
      self.record_action = True
      self.train_model(iteration)
      if self.hparams.patience>=0 and self.cur_attempt > self.hparams.patience: 
        break
      if iteration >= self.hparams.max_iter: 
        break
