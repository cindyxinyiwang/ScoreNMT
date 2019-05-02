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

from collections import defaultdict, deque
import copy
import time


class Featurizer():
  def __init__(self, hparams, data_loader):
    self.hparams = hparams
    self.data_loader = data_loader
    self.num_feature = self.hparams.lan_size

  def get_state(self, src, src_len, trg):
    existed_src = (np.array(src_len) > 2).astype(int).sum(axis=0)
    if self.hparams.feature_type == "lan_dist":
      src_dist = existed_src * self.data_loader.lan_dist_vec / 100
      #data_feature = np.append(src_dist, [iter_percent, cur_dev_ppl]).reshape(1, -1)
      #data_feature = np.append(src_dist, [iter_percent]).reshape(1, -1)
      data_feature = src_dist.reshape(1, -1)
      data_feature = Variable(torch.FloatTensor(data_feature))
    elif self.hparams.feature_type == "zero_one":
      data_feature = existed_src.reshape(1, -1)
      data_feature = Variable(torch.FloatTensor(data_feature))
    elif self.hparams.feature_type == "one":
      data_feature = Variable(torch.ones((1, len(existed_src)))) 

    existed_src = Variable(torch.FloatTensor([existed_src.tolist()]))
    if self.hparams.cuda: 
      data_feature = data_feature.cuda()
      existed_src = existed_src.cuda()
    return [data_feature, existed_src]

class EmbFeaturizer():
  def __init__(self, hparams, src_emb, trg_emb, data_loader):
    self.hparams = hparams
    self.data_loader = data_loader
    self.num_feature = self.hparams.lan_size
    self.src_emb = src_emb
    self.trg_emb = trg_emb

  def get_state(self, src, src_len, trg):
    existed_src = (np.array(src_len) > 2).astype(int).sum(axis=0)
    # special case of batch 1
    src = src[0]
    trg = trg[0]
    x, x_mask, x_count, x_len, x_pos_emb_idxs, _, _ = self.data_loader._pad(src, self.hparams.pad_id)
    y = Variable(torch.LongTensor(trg))
    x_len = Variable(torch.FloatTensor(x_len))
    y_len = Variable(torch.FloatTensor([len(trg)]))
    existed_src = Variable(torch.FloatTensor([existed_src.tolist()]))
    if self.hparams.cuda: 
      existed_src = existed_src.cuda()
      x_len = x_len.cuda()
      y = y.cuda()
      y_len = y_len.cuda()
    
    #1,lan_size,  dim
    x_emb = self.src_emb(x).sum(dim=1).view(1, -1, self.hparams.d_word_vec) / x_len.view(-1, 1)
    #1, 1, dim
    y_emb = self.src_emb(y).sum(dim=0).view(1, 1, -1) / y_len
    return ([x_emb, y_emb], existed_src)


