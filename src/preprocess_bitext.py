import argparse

import numpy as np
import random
import torch
import os
import gc

random.seed(0)
parser = argparse.ArgumentParser()

parser.add_argument("--save_data", type=str, help="path to saved data dir and the prefix")
parser.add_argument("--train_src", type=str, help="path to source file")
parser.add_argument("--train_trg", type=str, help="path to target file")
parser.add_argument("--dev_src", type=str, help="path to source file")
parser.add_argument("--dev_trg", type=str, help="path to target file")
parser.add_argument("--test_src", type=str, help="path to source file")
parser.add_argument("--test_trg", type=str, help="path to target file")
parser.add_argument("--src_vocab", type=str, help="path to source vocab file")
parser.add_argument("--trg_vocab", type=str, help="path to target vocab file")
parser.add_argument("--src_vocab_size", type=int, default=None, help="max size of source vocab")
parser.add_argument("--trg_vocab_size", type=int, default=None, help="max size of target vocab")
parser.add_argument("--src_char_vocab", type=str, default=None, help="path to source vocab file")
parser.add_argument("--trg_char_vocab", type=str, default=None, help="path to source vocab file")
parser.add_argument("--src_char_vocab_size", type=int, default=None, help="max size of source vocab")
parser.add_argument("--trg_char_vocab_size", type=int, default=None, help="max size of source vocab")
parser.add_argument("--max_len", type=int, default=0, help="max len filter for training")
parser.add_argument("--n", type=int, default=None, help="ngram n")
parser.add_argument("--ordered_char_dict", action="store_true", help="whether to use a combined char vocab for source")
parser.add_argument("--combined_src_char_vocab", action="store_true", help="whether to use a combined char vocab for source")
parser.add_argument("--combined_trg_word_vocab", type=str, default=None, help="whether to use a shared word vocab for target")
parser.add_argument("--langs", type=str, default="langs.txt", help="file of all language codes we use")


parser.add_argument("--shuffle", type=int, default=1, help="shuffle data")

args = parser.parse_args()

args.pad = "<pad>"
args.unk = "<unk>"
args.bos = "<s>"
args.eos = "<\s>"
args.pad_id = 0
args.unk_id = 1
args.bos_id = 2
args.eos_id = 3

if args.combined_trg_word_vocab:
  trg_word_file = args.combined_trg_word_vocab + ".pt"
  print("using vocab trg word from {}".format(trg_word_file))
  if not os.path.isfile(trg_word_file):
    trg_word_i2w = []
    trg_word_w2i = {}
    i = 0
    with open(args.combined_trg_word_vocab, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        if i == 0 and w != "<pad>":
          trg_word_i2w = ['<pad>', '<unk>', '<s>', '<\s>']
          trg_word_w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
          i = 4
        trg_word_w2i[w] = i
        trg_word_i2w.append(w)
        i += 1
        if args.trg_vocab_size and i >= arg.trg_vocab_size:
          break
    torch.save({"i2w": trg_word_i2w, "w2i": trg_word_w2i}, trg_word_file)
  else:
    trg_word_vocab = torch.load(trg_word_file)
    trg_word_i2w, trg_word_w2i = trg_word_vocab["i2w"], trg_word_vocab["w2i"]
  print("trg word vocab size {}".format(len(trg_word_i2w)))

if args.combined_src_char_vocab:
  vocab_suf = ""
  args.ordered_char_dict: vocab_suf = "o"
  if args.n == 4:
    vocab_suf += "char4vocab"
  elif args.n == 5:
    vocab_suf += "char5vocab"
  src_char_file = "data/" + args.langs.split(".")[0] + "-{}-char_vocab.pt".format(vocab_suf)
  print("using src char vocab from {}".format(src_char_file))
  if not os.path.isfile(src_char_file):
    src_char_w2i = {args.pad: args.pad_id, args.unk: args.unk_id, args.bos: args.bos_id, args.eos: args.eos_id}
    src_char_i2w = [args.pad_id, args.unk_id, args.bos_id, args.eos_id]
    # combine vocabs
    with open(args.langs, 'r') as myfile:
      for line in myfile:
        lan = line.strip()
        with open("data/{}_eng/ted-train.mtok.{}.{}".format(lan, lan, vocab_suf), "r") as vocab_file:
          for i, line in enumerate(vocab_file):
            w = line.strip()
            if w not in src_char_w2i: 
              src_char_w2i[w] = len(src_char_w2i)
              src_char_i2w.append(w)
            if args.src_char_vocab_size and i >= args.src_char_vocab_size: break
    torch.save({"i2w": src_char_i2w, "w2i": src_char_w2i}, src_char_file)
  else:
    src_char_vocab = torch.load(src_char_file)
    src_char_i2w = src_char_vocab["i2w"]
    src_char_w2i = src_char_vocab["w2i"]
  print("src char vocab size {}".format(len(src_char_i2w)))

class DataUtil(object):

  def __init__(self, hparams, shuffle=True):
    self.hparams = hparams
    
    self.shuffle = shuffle
    self.src_i2w, self.src_w2i = self._build_vocab(self.hparams.src_vocab, max_vocab_size=self.hparams.src_vocab_size)
    if args.combined_trg_word_vocab:
      self.trg_i2w, self.trg_w2i = trg_word_i2w, trg_word_w2i
    else:
      self.trg_i2w, self.trg_w2i = self._build_vocab(self.hparams.trg_vocab, max_vocab_size=self.hparams.trg_vocab_size)

    self.src_char_i2w, self.src_char_w2i, self.trg_char_i2w, self.trg_char_w2i = None, None, None, None
    if self.hparams.combined_src_char_vocab:
      self.src_char_i2w, self.src_char_w2i = src_char_i2w, src_char_w2i
    elif self.hparams.src_char_vocab:
      self.src_char_i2w, self.src_char_w2i = self._build_char_vocab_from(self.hparams.src_char_vocab, self.hparams.src_char_vocab_size, n=self.hparams.n)
    if self.hparams.trg_char_vocab:
      self.trg_char_i2w, self.trg_char_w2i = self._build_char_vocab_from(self.hparams.trg_char_vocab, self.hparams.trg_char_vocab_size, n=self.hparams.n)
    self.train_x, self.train_y, self.x_char_kv, self.y_char_kv, src_len = self._build_parallel(self.hparams.train_src, self.hparams.train_trg, shuffle=self.shuffle)
      
    self.dev_x, self.dev_y, self.dev_x_char_kv, self.dev_y_char_kv, src_len = self._build_parallel(self.hparams.dev_src, self.hparams.dev_trg, is_train=False)
    self.test_x, self.test_y, self.test_x_char_kv, self.test_y_char_kv, src_len = self._build_parallel(self.hparams.test_src, self.hparams.test_trg, is_train=False)

  def _build_parallel(self, src_file_name, trg_file_name, is_train=True, shuffle=True):
    print("loading parallel sentences from {} {} ".format(src_file_name, trg_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    src_char_kv_data = []
    trg_char_kv_data = []
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0

    src_lens = []
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = src_line.split()
      trg_tokens = trg_line.split()
      if is_train and (not src_tokens or not trg_tokens): 
        skip_line_count += 1
        continue
      if is_train and self.hparams.max_len and len(src_tokens) > self.hparams.max_len and len(trg_tokens) > self.hparams.max_len:
        skip_line_count += 1
        continue
      
      src_lens.append(len(src_tokens))
      src_indices, trg_indices = [self.hparams.bos_id], [self.hparams.bos_id] 
      if self.hparams.src_char_vocab:
          src_char_kv = [{0:0}]
      if self.hparams.trg_char_vocab:
          trg_char_kv = [{0:0}]
      src_w2i = self.src_w2i
      for src_tok in src_tokens:
        #print(src_tok)
        if src_tok not in src_w2i:
          src_indices.append(self.hparams.unk_id)
          src_unk_count += 1
          #print("unk {}".format(src_unk_count))
        else:
          src_indices.append(src_w2i[src_tok])
          #print("src id {}".format(src_w2i[src_tok]))
        # calculate char ngram emb for src_tok
        if self.hparams.src_char_vocab:
          ngram_counts = self._get_ngram_counts(src_tok, self.src_char_i2w, self.src_char_w2i, self.hparams.n)
          src_char_kv.append(ngram_counts)

      trg_w2i = self.trg_w2i
      for trg_tok in trg_tokens:
        if trg_tok not in trg_w2i:
          trg_indices.append(self.hparams.unk_id)
          trg_unk_count += 1
        else:
          trg_indices.append(trg_w2i[trg_tok])
        # calculate char ngram emb for trg_tok
        if self.hparams.trg_char_vocab:
          ngram_counts = self._get_ngram_counts(trg_tok, self.trg_char_i2w, self.trg_char_w2i, self.hparams.n)
          trg_char_kv.append(ngram_counts)

      src_indices.append(self.hparams.eos_id)
      trg_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      trg_data.append(trg_indices)
      if self.hparams.src_char_vocab:
        src_char_kv.append({0:0})
        src_char_kv_data.append(src_char_kv)
      if self.hparams.trg_char_vocab:
        trg_char_kv.append({0:0})
        trg_char_kv_data.append(trg_char_kv)
      line_count += 1
      if line_count % 10000 == 0:
        print("processed {} lines".format(line_count))
    if is_train and shuffle:
      if self.hparams.src_char_vocab :
        src_data, trg_data, src_char_kv_data, trg_char_kv_data = self.sort_by_xlen([src_data, trg_data, src_char_kv_data, trg_char_kv_data], descend=False)
      else:
        src_data, trg_data = self.sort_by_xlen([src_data, trg_data], descend=False)
    print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
    assert len(src_data) == len(trg_data)
    print("lines={}, skipped_lines={}".format(len(src_data), skip_line_count))

    return src_data, trg_data, src_char_kv_data, trg_char_kv_data, src_lens

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

    return i2w, w2i


  def _build_char_vocab_from(self, vocab_file, vocab_size, n=None,
      single_n=False):
    vocab_size = int(vocab_size)
    i2w = [ '<unk>']
    i2w_set = set(i2w) 
    cur_vsize = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        if single_n and n and len(w) != n: continue
        if not single_n and n and len(w) > n: continue 
        if w == '<unk>' or w == '<pad>' or w == '<s>' or w == '<\s>': continue
        if w not in i2w_set:
          cur_vsize += 1
          i2w.append(w)
          i2w_set.add(w)
          if vocab_size and cur_vsize > vocab_size: break
    w2i = {}
    for i, w in enumerate(i2w):
      w2i[w] = i
    return i2w, w2i

  def sort_by_xlen(self, data_list, descend=True):
    array_list = [np.array(x) for x in data_list]
    x_len = [len(i) for i in data_list[0]]
    index = np.argsort(x_len)
    if descend:
      index = index[::-1]
    for i, x in enumerate(array_list):
      if not x is None:
        data_list[i] = x[index].tolist()
    return data_list 

  def _get_ngram_counts(self, word, i2w, w2i, n):
    count = {}
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

if __name__ == "__main__":
  save_data = args.save_data
  train, dev, test = {}, {}, {}
  if args.combined_src_char_vocab:
    with open(args.langs, "r") as myfile:
      for line in myfile:
        gc.collect()
        train, dev, test = {}, {}, {}
        lan = line.strip()
        args.save_data = save_data.format(lan)
        args.train_src = "data/{}_eng/ted-train.mtok.{}".format(lan, lan)
        args.src_vocab = "data/{}_eng/ted-train.mtok.{}.vocab".format(lan, lan)
        args.dev_src = "data/{}_eng/ted-dev.mtok.{}".format(lan, lan)
        args.test_src = "data/{}_eng/ted-test.mtok.{}".format(lan, lan)
        args.train_trg = "data/{}_eng/ted-train.mtok.spm8000.eng".format(lan)
        args.dev_trg = "data/{}_eng/ted-dev.mtok.spm8000.eng".format(lan)
        args.test_trg = "data/{}_eng/ted-test.mtok.spm8000.eng".format(lan)
        data = DataUtil(hparams=args, shuffle=args.shuffle)
        train["train_y"] = data.train_y
        train["x_char_kv"] = data.x_char_kv

        dev["dev_y"] = data.dev_y
        dev["dev_x_char_kv"] = data.dev_x_char_kv
        
        test["test_y"] = data.test_y
        test["test_x_char_kv"] = data.test_x_char_kv
        print("save data to {}...".format(args.save_data))
        torch.save(train, args.save_data + "-train.pt")
        torch.save(dev, args.save_data + "-dev.pt")
        torch.save(test, args.save_data + "-test.pt")
  else:
    train["train_y"] = data.train_y
    train["train_x"] = data.train_x

    dev["dev_y"] = data.dev_y
    dev["dev_x"] = data.dev_x
    
    test["test_y"] = data.test_y
    test["test_x"] = data.test_x
    vocab = {}
    vocab["src_w2i"] = data.src_w2i
    vocab["src_char_w2i"] = data.src_char_w2i
    vocab["src_i2w"] = data.src_i2w
    vocab["src_char_i2w"] = data.src_char_i2w
    vocab["trg_w2i"] = data.trg_w2i
    vocab["trg_char_w2i"] = data.trg_char_w2i
    vocab["trg_i2w"] = data.trg_i2w
    vocab["trg_char_i2w"] = data.trg_char_i2w
 

