import random
import numpy as np
import os
import functools

import torch
from torch.autograd import Variable
# multilingual data utils


class ClassDataUtil(object):
  def __init__(self, hparams, shuffle=True):
    self.hparams = hparams
    self.src_i2w_list = []
    self.src_w2i_list = []
    
    self.shuffle = shuffle

    if self.hparams.src_vocab:
      self.src_i2w, self.src_w2i = self._build_vocab(self.hparams.src_vocab, max_vocab_size=self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
    else:
      print("not using single src word vocab..")

    if self.hparams.lang_file:
      self.train_src_file_list = []
      self.dev_src_file_list = []
      if self.hparams.src_char_vocab_from:
        self.src_char_vocab_from = []
      if self.hparams.src_vocab_list:
        self.src_vocab_list = []
      self.lans = []
      with open(self.hparams.lang_file, "r") as myfile:
        for line in myfile:
          lan = line.strip()
          self.lans.append(lan)
          if self.hparams.src_char_vocab_from:
            self.src_char_vocab_from.append(self.hparams.src_char_vocab_from.replace("LAN", lan))
          self.train_src_file_list.append(self.hparams.train_src_file_list[0].replace("LAN", lan))
          self.dev_src_file_list.append(self.hparams.dev_src_file_list[0].replace("LAN", lan))
          if self.hparams.src_vocab_list:
            self.src_vocab_list.append(self.hparams.src_vocab_list[0].replace("LAN", lan))
      self.hparams.lan_size = len(self.train_src_file_list)
    self.hparams.trg_vocab_size = self.hparams.lan_size
    if self.hparams.src_vocab_list:
      self.src_i2w, self.src_w2i = self._build_char_vocab_from(self.src_vocab_list, self.hparams.src_vocab_size)
      self.hparams.src_vocab_size = len(self.src_i2w)
      print("use combined src vocab at size {}".format(self.hparams.src_vocab_size))

    if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
      self.src_char_i2w, self.src_char_w2i = self._build_char_vocab_from(self.src_char_vocab_from, self.hparams.src_char_vocab_size, n=self.hparams.char_ngram_n)
      self.src_char_vsize = len(self.src_char_i2w)
      setattr(self.hparams, 'src_char_vsize', self.src_char_vsize)
      print("src_char_vsize={}".format(self.src_char_vsize))
    else:
      self.src_char_vsize = None
      setattr(self.hparams, 'src_char_vsize', None)

    if not self.hparams.decode:
      self.start_indices = []
      self.end_indices = []
      self.x_train = []
      self.x_char_kv = []
      self.x_len = []
      self.y_train = []
      for data_idx in range(len(self.train_src_file_list)):
        x_train, y_train, x_char_kv, x_len = self._build_parallel(self.train_src_file_list[data_idx], data_idx, outprint=True)
        self.x_train.extend(x_train)
        self.x_char_kv.extend(x_char_kv)
        self.x_len.extend(x_len)
        self.y_train.extend(y_train)
      if self.shuffle:
        self.x_train, self.y_train, self.x_char_kv, self.x_len = self.sort_by_xlen([self.x_train, self.y_train, self.x_char_kv, self.x_len], descend=False)
      # set batcher indices once
      self.start_indices, self.end_indices = [], []
      if self.hparams.batcher == "word":
        start_index, end_index, count = 0, 0, 0
        while True:
          count += x_len[end_index]
          end_index += 1
          if end_index >= len(self.x_len):
            self.start_indices.append(start_index)
            self.end_indices.append(end_index)
            break
          if count > self.hparams.batch_size: 
            self.start_indices.append(start_index)
            self.end_indices.append(end_index)
            count = 0
            start_index = end_index
      elif self.hparams.batcher == "sent":
        start_index, end_index, count = 0, 0, 0
        while end_index < len(self.x_len):
          end_index = min(start_index + self.hparams.batch_size, len(self.x_len))
          self.start_indices.append(start_index)
          self.end_indices.append(end_index)
          start_index = end_index
      else:
        print("unknown batcher")
        exit(1)

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
    return ret

  def next_train(self):
    while True:
      step = 0
      for step_b, batch_idx in enumerate(np.random.permutation(len(self.start_indices))):
        step += 1
        yield self.yield_data(batch_idx, self.x_train, self.x_char_kv, self.y_train, step_b)
 
  def yield_data(self, batch_idx, x_train, x_char_kv, y_train, step):
    start_index, end_index = self.start_indices[batch_idx], self.end_indices[batch_idx]
    x, y, x_char = [], [], [] 
    if x_train:
      x = x_train[start_index:end_index]
    if x_char_kv:
      x_char = x_char_kv[start_index:end_index]
    y = y_train[start_index:end_index]
    train_file_index = y_train 
    if self.shuffle:
      x, y, x_char, train_file_index = self.sort_by_xlen([x, y, x_char, train_file_index])

    # pad
    x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize)
    batch_size = end_index - start_index
    if step == len(self.start_indices)-1:
      eop = True
    else:
      eop = False

    return x, x_mask, x_count, x_len, x_pos_emb_idxs, y, batch_size, x_char, None, eop, train_file_index

  def next_dev(self, dev_batch_size=1):
    first_dev = True
    while True:
      for data_idx in range(len(self.dev_src_file_list)):
        x_dev, y_dev, x_char_kv, x_dev_len = self._build_parallel(self.dev_src_file_list[data_idx], data_idx, is_train=False, outprint=first_dev)
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
          if self.shuffle:
            x, y, x_char, dev_file_index = self.sort_by_xlen([x, y, x_char, dev_file_index])

          # pad
          x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize)
          batch_size = end_index - start_index
          if end_index == len(x_dev_len):
            eof = True
          else:
            eof = False
          if data_idx == len(self.dev_src_file_list)-1 and eof:
            eop = True
          else:
            eop = False
          start_index = end_index
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y,  batch_size, x_char, None, eop, eof, dev_file_index


  def next_test(self, test_batch_size=1):
    while True:
      for data_idx in range(len(self.hparams.test_src_file_list)):
        x_test, y_test, x_char_kv, x_test_len = self._build_parallel(self.hparams.test_src_file_list[data_idx], self.hparams.test_file_idx_list[data_idx], is_train=False, outprint=True)
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
          x, x_mask, x_count, x_len, x_pos_emb_idxs, x_char = self._pad(x, self.hparams.pad_id, x_char, self.hparams.src_char_vsize)
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
          yield x, x_mask, x_count, x_len, x_pos_emb_idxs, y, batch_size, x_char, None, eop, eof, test_file_index


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

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None):
    if sentences:
      batch_size = len(sentences)
      lengths = [len(s) for s in sentences]
      count = sum(lengths)
      max_len = max(lengths)
      padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]
      padded_sentences = Variable(torch.LongTensor(padded_sentences))
      char_sparse = None
    else:
      batch_size = len(char_kv)
      lengths = [len(s) for s in char_kv]
      padded_sentences = None
      count = sum(lengths)
      max_len = max(lengths)
      char_sparse = []
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
    return padded_sentences, mask, count, lengths, pos_emb_indices, char_sparse

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


  def _build_parallel(self, src_file_name, trg_idx, is_train=True, shuffle=True, outprint=False):
    if outprint:
      print("loading parallel sentences from {} ".format(src_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    src_char_kv_data = []
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0

    src_lens = []

    for src_line in src_lines:
      src_tokens = src_line.split()
      if is_train and not src_tokens: 
        skip_line_count += 1
        continue
      if is_train and not self.hparams.decode and self.hparams.max_len and len(src_tokens) > self.hparams.max_len:
        skip_line_count += 1
        continue
      
      src_lens.append(len(src_tokens))
      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_char_kv = [{0:0}]
      else:
        src_indices = [self.hparams.bos_id] 
      for src_tok in src_tokens:
        # calculate char ngram emb for src_tok
        if self.hparams.char_ngram_n > 0:
          ngram_counts = self._get_ngram_counts(src_tok)
          src_char_kv.append(ngram_counts)
        elif self.hparams.bpe_ngram:
          ngram_counts = self._get_bpe_ngram_counts(src_tok, self.src_char_i2w, self.src_char_w2i)
          src_char_kv.append(ngram_counts)
        else:
          if src_tok not in self.src_w2i:
            src_indices.append(self.hparams.unk_id)
            src_unk_count += 1
          else:
            src_indices.append(self.src_w2i[src_tok])

      if self.hparams.char_ngram_n > 0 or self.hparams.bpe_ngram:
        src_char_kv.append({0:0})
        src_char_kv_data.append(src_char_kv)
      else:
        src_indices.append(self.hparams.eos_id)
        src_data.append(src_indices)
      trg_data.append(trg_idx)
      line_count += 1
      if outprint:
        if line_count % 10000 == 0:
          print("processed {} lines".format(line_count))

    if is_train and shuffle:
      src_data, trg_data, src_char_kv_data = self.sort_by_xlen([src_data, trg_data, src_char_kv_data], descend=False)
    if outprint:
      print("src_unk={}".format(src_unk_count))
      print("lines={}, skipped_lines={}".format(len(trg_data), skip_line_count))
    return src_data, trg_data, src_char_kv_data, src_lens

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

