import random
import numpy as np
import os
import functools
import subprocess

import torch
from torch.autograd import Variable
# multilingual data utils


class MultDataUtil(object):
  def __init__(self, hparams, shuffle=True):
    self.hparams = hparams
    self.src_i2w_list = []
    self.src_w2i_list = []
    
    self.shuffle = shuffle
    self.record_action = True

    if self.hparams.src_vocab:
      self.src_i2w, self.src_w2i = self._build_vocab(self.hparams.src_vocab, max_vocab_size=self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
    else:
      print("not using single src word vocab..")

    if self.hparams.trg_vocab:
      self.trg_i2w, self.trg_w2i = self._build_vocab(self.hparams.trg_vocab, max_vocab_size=self.hparams.trg_vocab_size)
      self.hparams.trg_vocab_size = len(self.trg_i2w)
    else:
      print("not using single trg word vocab..")

    self.total_train_count = 0
    if self.hparams.lang_file:
      self.train_src_file_list = []
      self.train_trg_file_list = []
      if self.hparams.src_char_vocab_from:
        self.src_char_vocab_from = []
      if self.hparams.src_vocab_list:
        self.src_vocab_list = []
      self.lan_i2w = []
      self.lan_w2i = {} 
      with open(self.hparams.lang_file, "r") as myfile:
        i = 0
        for line in myfile:
          lan = line.strip()
          self.lan_i2w.append(lan)
          self.lan_w2i[lan] = i
          i += 1
          if self.hparams.src_char_vocab_from:
            self.src_char_vocab_from.append(self.hparams.src_char_vocab_from.replace("LAN", lan))
          self.train_src_file_list.append(self.hparams.train_src_file_list[0].replace("LAN", lan))
          self.train_trg_file_list.append(self.hparams.train_trg_file_list[0].replace("LAN", lan))
          
          train_line_count = int(subprocess.getoutput("wc -l {0}".format(self.train_src_file_list[-1])).split()[0])
          self.total_train_count += train_line_count

          if self.hparams.src_vocab_list:
            self.src_vocab_list.append(self.hparams.src_vocab_list[0].replace("LAN", lan))

      self.hparams.lan_size = len(self.train_src_file_list)
      self.get_lan_dist()
    if self.hparams.src_vocab_list:
      self.src_i2w, self.src_w2i = self._build_char_vocab_from(self.src_vocab_list, self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
      print("use combined src vocab at size {}".format(self.hparams.src_vocab_size))

    if self.hparams.src_char_vocab_from:
      self.src_char_i2w, self.src_char_w2i = self._build_char_vocab_from(self.src_char_vocab_from, self.hparams.src_char_vocab_size, n=self.hparams.char_ngram_n, single_n=self.hparams.single_n)
      self.src_char_vsize = len(self.src_char_i2w)
      setattr(self.hparams, 'src_char_vsize', self.src_char_vsize)
      print("src_char_vsize={}".format(self.src_char_vsize))
      #if self.hparams.compute_ngram:
      #  self.
    else:
      self.src_char_vsize = None
      setattr(self.hparams, 'src_char_vsize', None)

    if (not self.hparams.decode):
      self.start_indices = [[] for i in range(len(self.train_src_file_list))]
      self.end_indices = [[] for i in range(len(self.train_src_file_list))]
      self.actions = [[] for i in range(len(self.train_src_file_list))]

  def get_lan_dist(self):
    self.lan_dists = [[-1 for _ in range(self.hparams.lan_size)] for _ in range(self.hparams.lan_size)]
    with open(self.hparams.lan_dist_file, 'r') as myfile:
      for line in myfile:
        toks = line.split()
        if toks[0] in self.lan_w2i and toks[1] in self.lan_w2i:
          l1, l2 = self.lan_w2i[toks[0]], self.lan_w2i[toks[1]]
          self.lan_dists[l1][l2] = float(toks[2])

  def query_lan_dist(self, id_1, id_2):
    return self.lan_dists[id_1][id_2]

  def get_char_emb(self, word_idx, is_trg=True):
    if is_trg:
      w2i, i2w, vsize = self.trg_char_w2i, self.trg_char_i2w, self.hparams.trg_char_vsize
      word = self.trg_i2w_list[0][word_idx]
    else:
      w2i, i2w, vsize = self.src_char_w2i, self.src_char_i2w, self.hparams.src_char_vsize
      word = self.src_i2w_list[0][word_idx]
    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      if word_idx == self.hparams.bos_id or word_idx == self.hparams.eos_id:
        kv = {0:0}
      elif self.hparams.char_ngram_n:
        kv = self._get_ngram_counts(word, i2w, w2i, self.hparams.char_ngram_n)
      elif self.hparams.bpe_ngram:
        kv = self._get_bpe_ngram_counts(word, i2w, w2i)
      key = torch.LongTensor([[0 for _ in range(len(kv.keys()))], list(kv.keys())])
      val = torch.FloatTensor(list(kv.values()))
      ret = [torch.sparse.FloatTensor(key, val, torch.Size([1, vsize]))]
    elif self.hparams.char_input is not None:
      ret = self._get_char(word, i2w, w2i, n=self.hparams.n)
      ret = Variable(torch.LongTensor(ret).unsqueeze(0).unsqueeze(0))
      if self.hparams.cuda: ret = ret.cuda()
    return ret

  def get_base_data(self):
    x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[0], self.train_trg_file_list[0], 0, outprint=True, load_full=True)
    return x_train, x_char_kv, y_train

  def next_train(self, load_full=True, log=True):
    while True:
      step = -1
      if self.hparams.lang_shuffle:
        self.train_data_queue = np.random.permutation(len(self.train_src_file_list))
      else:
        self.train_data_queue = [i for i in range(len(self.train_src_file_list))]
      for data_idx in self.train_data_queue:
        x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[data_idx], self.train_trg_file_list[data_idx], data_idx, outprint=(log), sample=self.hparams.subsample, load_full=load_full)
        #x_train, y_train, x_char_kv, x_len, x_rank = self._build_parallel(self.train_src_file_list[data_idx], self.train_trg_file_list[data_idx], outprint=True)
        # set batcher indices once
        if len(x_len) == 0:
          print("skipping 0 size file {}".format(self.train_src_file_list[data_idx]))
          continue
        #if not self.start_indices[data_idx]:
        if True:
          start_indices, end_indices = [], []
          if self.hparams.batcher == "word":
            start_index, end_index, count = 0, 0, 0
            while True:
              count += (x_len[end_index] + len(y_train[end_index]))
              end_index += 1
              if end_index >= len(x_len):
                start_indices.append(start_index)
                end_indices.append(end_index)
                break
              if count > self.hparams.batch_size:
                start_indices.append(start_index)
                end_indices.append(end_index)
                count = 0
                start_index = end_index
              
          elif self.hparams.batcher == "sent":
            start_index, end_index, count = 0, 0, 0
            while end_index < len(x_len):
              end_index = min(start_index + self.hparams.batch_size, len(x_len))
              start_indices.append(start_index)
              end_indices.append(end_index)
              start_index = end_index
          else:
            print("unknown batcher")
            exit(1)
          self.start_indices[data_idx] = start_indices
          self.end_indices[data_idx] = end_indices
          if self.record_action:
            self.actions[data_idx] = [0 for _ in range(len(start_indices))]
        cached = []
        for step_b, batch_idx in enumerate(np.random.permutation(len(self.start_indices[data_idx]))):
          step += 1
          yield self.yield_data(data_idx, batch_idx, x_train, x_char_kv, y_train, step_b, x_rank)

  def update_action(self, data_idx, batch_idx, action):
    self.actions[data_idx][batch_idx] = action
 
  def yield_data(self, data_idx, batch_idx, x_train, x_char_kv, y_train, step, x_train_rank):
    start_index, end_index = self.start_indices[data_idx][batch_idx], self.end_indices[data_idx][batch_idx]
    x, y, x_char, x_rank = [], [], [], []
    if x_train:
      x = x_train[start_index:end_index]
    if x_char_kv:
      x_char = x_char_kv[start_index:end_index]
    if x_train_rank:
      x_rank = x_train_rank[start_index:end_index]

    y = y_train[start_index:end_index]
    train_file_index = [data_idx for i in range(end_index - start_index)] 
    if self.shuffle:
      x, y, x_char, train_file_index, x_rank = self.sort_by_xlen([x, y, x_char, train_file_index, x_rank])
    x_list, y_list = x, y
    # pad
    x, x_mask, x_count, x_len, x_pos_emb_idxs, _, x_rank = self._pad(x, self.hparams.pad_id, [], self.hparams.src_char_vsize, x_rank)
    y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
    batch_size = end_index - start_index
    if step == len(self.start_indices[data_idx])-1:
      eof = True
    else:
      eof = False
    if data_idx != 0 and data_idx == self.train_data_queue[-1] and step == len(self.start_indices[data_idx])-1:
      eop = True
    else:
      eop = False
    return x, x_list, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_list, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, train_file_index, x_rank, data_idx, batch_idx, self.actions[data_idx][batch_idx]

  def next_dev(self, dev_batch_size=1, data_idx=0, load_full=True, log=True):
    first_dev = True
    idxes = [0]
    if data_idx != 0:
      idxes.append(data_idx)
    else:
      idxes.append(1)
    if len(self.hparams.dev_src_file_list) == 1:
      idxes = [0]
    while True:
      #for data_idx in range(len(self.hparams.dev_src_file_list)):
      for data_idx in idxes:
        max_line = self.hparams.agent_dev_num
        if load_full: max_line = None
        x_dev, y_dev, x_char_kv, x_dev_len, x_dev_rank = self._build_parallel(self.hparams.dev_src_file_list[data_idx], self.hparams.dev_trg_file_list[data_idx], data_idx, is_train=False, outprint=log, sample=False, max_line=max_line, load_full=load_full)
        first_dev = False
        start_index, end_index = 0, 0
        while end_index < len(x_dev_len):
          end_index = min(start_index + dev_batch_size, len(x_dev_len))
          x, y, x_char = [], [], [] 
          if x_dev:
            x = x_dev[start_index:end_index]
          if x_char_kv:
            x_char = x_char_kv[start_index:end_index]
          y = y_dev[start_index:end_index]
          dev_file_index = [self.hparams.dev_file_idx_list[data_idx] for i in range(end_index - start_index)]
          x_rank = []
          if self.shuffle:
            x, y, x_char, dev_file_index, x_rank = self.sort_by_xlen([x, y, x_char, dev_file_index, x_rank])
          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, _, x_rank = self._pad(x, self.hparams.pad_id, [], self.hparams.src_char_vsize, x_rank)
          y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
          batch_size = end_index - start_index
          if end_index == len(x_dev_len):
            eof = True
          else:
            eof = False
          if data_idx == idxes[-1] and eof:
            eop = True
          else:
            eop = False
          start_index = end_index
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, dev_file_index, x_rank


  def next_test(self, test_batch_size=1):
    while True:
      for data_idx in range(len(self.hparams.test_src_file_list)):
        x_test, y_test, x_char_kv, x_test_len, x_rank = self._build_parallel(self.hparams.test_src_file_list[data_idx], self.hparams.test_trg_file_list[data_idx], data_idx, is_train=False, sample=False, outprint=True)
        start_index, end_index = 0, 0
        while end_index < len(x_test_len):
          end_index = min(start_index + test_batch_size, len(x_test_len))
          x, y, x_char = [], [], [] 
          if x_test:
            x = x_test[start_index:end_index]
          if x_char_kv:
            x_char = x_char_kv[start_index:end_index]
          y = y_test[start_index:end_index]
          test_file_index = [self.hparams.test_file_idx_list[data_idx] for i in range(end_index - start_index)] 
          if self.shuffle:
            x, y, x_char, test_file_index = self.sort_by_xlen([x, y, x_char, test_file_index])

          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, _, x_rank = self._pad(x, self.hparams.pad_id, [], self.hparams.src_char_vsize, x_rank)
          y, y_mask, y_count, y_len, y_pos_emb_idxs, y_char, y_rank = self._pad(y, self.hparams.pad_id)
          batch_size = end_index - start_index
          if end_index == len(x_test_len):
            eof = True
          else:
            eof = False
          if data_idx == len(self.hparams.test_src_file_list)-1 and eof:
            eop = True
          else:
            eop = False
          start_index = end_index
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, y_mask, y_count, y_len, y_pos_emb_idxs, batch_size, x_char, None, eop, eof, test_file_index, x_rank


  def sort_by_xlen(self, data_list, descend=True):
    array_list = [np.array(x) for x in data_list]
    if data_list[0]:
      x_len = [len(i) for i in data_list[0]]
    else:
      x_len = [len(i) for i in data_list[2]]
    index = np.argsort(x_len)
    if descend:
      index = index[::-1]
    for i, x in enumerate(array_list):
      if x is not None and len(x) > 0:
        data_list[i] = x[index].tolist()
    return data_list 

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None, x_rank=None):
    if sentences:
      batch_size = len(sentences)
      lengths = [len(s) for s in sentences]
      count = sum(lengths)
      max_len = max(lengths)
      padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]
      padded_sentences = Variable(torch.LongTensor(padded_sentences))
      char_sparse = None
      padded_x_rank = None
    else:
      batch_size = len(char_kv)
      lengths = [len(s) for s in char_kv]
      padded_sentences = None
      count = sum(lengths)
      max_len = max(lengths)
      char_sparse = []
      if x_rank:
        padded_x_rank = [s + ([0]*(max_len - len(s))) for s in x_rank]
      else:
        padded_x_rank = None

      for kvs in char_kv:
        sent_sparse = []
        key, val = [], []
        for i, kv in enumerate(kvs):
          key.append(torch.LongTensor([[i for _ in range(len(kv.keys()))], list(kv.keys())]))
          val.extend(list(kv.values()))
        key = torch.cat(key, dim=1)
        val = torch.FloatTensor(val)
        sent_sparse = torch.sparse.FloatTensor(key, val, torch.Size([max_len, char_dim]))
        # (batch_size, max_len, char_dim)
        char_sparse.append(sent_sparse)
    mask = [[0]*l + [1]*(max_len - l) for l in lengths]
    mask = torch.ByteTensor(mask)
    pos_emb_indices = [[i+1 for i in range(l)] + ([0]*(max_len - l)) for l in lengths]
    pos_emb_indices = Variable(torch.FloatTensor(pos_emb_indices))
    if self.hparams.cuda:
      if sentences:
        padded_sentences = padded_sentences.cuda()
      pos_emb_indices = pos_emb_indices.cuda()
      mask = mask.cuda()
    return padded_sentences, mask, count, lengths, pos_emb_indices, char_sparse, padded_x_rank

  def _get_char(self, word, i2w, w2i, n=1):
    chars = []
    for i in range(0, max(1, len(word)-n+1)):
      j = min(len(word), i+n)
      c = word[i:j]
      if c in w2i:
        chars.append(w2i[c])
      else:
        chars.append(self.hparams.unk_id)
    return chars

  @functools.lru_cache(maxsize=8000, typed=False)
  def _get_ngram_counts(self, word):
    count = {}
    for i in range(len(word)):
      for j in range(i+1, min(len(word), i+self.hparams.char_ngram_n)+1):
        ngram = word[i:j]
        if ngram in self.src_char_w2i:
          ngram = self.src_char_w2i[ngram]
        else:
          ngram = 0
        if ngram not in count: count[ngram] = 0
        count[ngram] += 1
    return count

  def _get_bpe_ngram_counts(self, word, i2w, w2i):
    count = {}
    word = "▁" + word
    n = len(word)
    for i in range(len(word)):
      for j in range(i+1, min(len(word), i+n)+1):
        ngram = word[i:j]
        if ngram in w2i:
          ngram = w2i[ngram]
        else:
          ngram = 0
        if ngram not in count: count[ngram] = 0
        count[ngram] += 1
    return count


  def _build_parallel(self, src_file_name, trg_file_name, data_idx, is_train=True, shuffle=True, outprint=False, sample=False, max_line=None, load_full=True):
    if outprint:
      print("loading parallel sentences from {} {}".format(src_file_name, trg_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')

    src_char_kv_data = []
    src_data = []
    trg_data = []

    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0
    src_word_rank = [] 

    src_lens = []
    line_n = -1
    if is_train and not load_full:
      if self.hparams.agent_subsample_percent:
        max_line = int(self.hparams.agent_subsample_percent * len(src_lines))
      elif self.hparams.agent_subsample_line:
        max_line = self.hparams.agent_subsample_line
      else:
        print("error subsample line not set")
        exit(1)
    if max_line is not None:
      if sample:
        indices = np.random.permutation(np.arange(len(src_lines)))[:max_line]
        src_lines = np.array(src_lines)[indices].tolist()
        trg_lines = np.array(trg_lines)[indices].tolist()
      else:
        src_lines = src_lines[:max_line]
        trg_lines = trg_lines[:max_line]
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      line_n += 1
      if is_train and not src_tokens or not trg_tokens: 
        skip_line_count += 1
        continue
      if is_train and not self.hparams.decode and self.hparams.max_len and (len(src_tokens) > self.hparams.max_len or len(trg_tokens) > self.hparams.max_len):
        skip_line_count += 1
        continue
      src_lens.append(len(src_tokens))
      src_indices = [self.hparams.bos_id]
      trg_indices = [self.hparams.bos_id]
      if self.hparams.src_char_vocab_from:
        src_char_kv = [{0:0}]
      src_indices = [self.hparams.bos_id] 
      for src_tok in src_tokens:
        # calculate char ngram emb for src_tok
        if self.hparams.src_char_vocab_from:
          ngram_counts = self._get_ngram_counts(src_tok)
          src_char_kv.append(ngram_counts)

        if src_tok not in self.src_w2i:
          src_indices.append(self.hparams.unk_id)
          src_unk_count += 1
        else:
          src_indices.append(self.src_w2i[src_tok])

      for trg_tok in trg_tokens:
        if trg_tok not in self.trg_w2i:
          trg_indices.append(self.hparams.unk_id)
          trg_unk_count += 1
        else:
          trg_indices.append(self.trg_w2i[trg_tok])

      trg_indices.append(self.hparams.eos_id)
      trg_data.append(trg_indices)
      if self.hparams.src_char_vocab_from:
        src_char_kv.append({0:0})
        src_char_kv_data.append(src_char_kv)
      src_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      line_count += 1
      #if line_count == 20: break
      #if (not max_line is None) and line_count > max_line: break 
      #if is_train and not load_full:
      #  if line_count > agent_subsample_max_line: break 

      if outprint:
        if line_count % 10000 == 0:
          print("processed {} lines".format(line_count))

    if is_train and shuffle:
      src_data, trg_data, src_char_kv_data, src_word_rank = self.sort_by_xlen([src_data, trg_data, src_char_kv_data, src_word_rank], descend=False)
    if outprint:
      print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
      print("lines={}, skipped_lines={}".format(len(trg_data), skip_line_count))
    return src_data, trg_data, src_char_kv_data, src_lens, src_word_rank

  def _build_char_vocab(self, lines, n=1):
    i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    for line in lines:
      words = line.split()
      for w in words:
        for i in range(0, max(1, len(w)-n+1)):
        #for c in w:
          j = min(len(w), i+n)
          c = w[i:j]
          if c not in w2i:
            w2i[c] = len(w2i)
            i2w.append(c)
    return i2w, w2i

  def _build_char_ngram_vocab(self, lines, n, max_char_vocab_size=None):
    i2w = ['<unk>']
    w2i = {}
    w2i['<unk>'] = 0

    for line in lines:
      words = line.split()
      for w in words:
        for i in range(len(w)):
          for j in range(i+1, min(i+n, len(w))+1):
            char = w[i:j]
            if char not in w2i:
              w2i[char] = len(w2i)
              i2w.append(char)
              if max_char_vocab_size and len(i2w) >= max_char_vocab_size: 
                return i2w, w2i
    return i2w, w2i

  def _build_vocab(self, vocab_file, max_vocab_size=None):
    i2w = []
    w2i = {}
    i = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        if i == 0 and w != "<pad>":
          i2w = ['<pad>', '<unk>', '<s>', '<\s>']
          w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
          i = 4
        w2i[w] = i
        i2w.append(w)
        i += 1
        if max_vocab_size and i >= max_vocab_size:
          break
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i

  def _build_vocab_list(self, vocab_file_list, max_vocab_size=None):
    i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    i = 4
    for vocab_file in vocab_file_list:
      with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
          w = line.strip()
          if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
          w2i[w] = i
          i2w.append(w)
          i += 1
          if max_vocab_size and i >= max_vocab_size:
            break
    assert i2w[self.hparams.pad_id] == '<pad>'
    assert i2w[self.hparams.unk_id] == '<unk>'
    assert i2w[self.hparams.bos_id] == '<s>'
    assert i2w[self.hparams.eos_id] == '<\s>'
    assert w2i['<pad>'] == self.hparams.pad_id
    assert w2i['<unk>'] == self.hparams.unk_id
    assert w2i['<s>'] == self.hparams.bos_id
    assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i


  def _build_char_vocab_from(self, vocab_file_list, vocab_size_list, n=None,
      single_n=False):
    vfile_list = vocab_file_list
    if type(vocab_file_list) != list:
      vsize_list = [int(s) for s in vocab_size_list.split(",")]
    elif not vocab_size_list:
      vsize_list = [0 for i in range(len(vocab_file_list))]
    else:
      vsize_list = [int(vocab_size_list) for i in range(len(vocab_file_list))]
    while len(vsize_list) < len(vfile_list):
      vsize_list.append(vsize_list[-1])
    #if self.hparams.ordered_char_dict:
    i2w = [ '<unk>']
    i2w_set = set(i2w)
    for vfile, size in zip(vfile_list, vsize_list):
      cur_vsize = 0
      with open(vfile, 'r', encoding='utf-8') as f:
        for line in f:
          w = line.strip()
          if single_n and n and len(w) != n: continue
          if not single_n and n and len(w) > n: continue 
          if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
          cur_vsize += 1
          if w not in i2w_set:
            i2w.append(w)
            i2w_set.add(w)
            if size > 0 and cur_vsize > size: break
    w2i = {}
    for i, w in enumerate(i2w):
      w2i[w] = i
    return i2w, w2i

