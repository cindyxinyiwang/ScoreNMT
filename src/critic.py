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

class Critic(nn.Module):
  def __init__(self, hparams, num_feature):
    super(Critic, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden_critic
    self.w = nn.Linear(num_feature, hidden_size, bias=False)
    self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
    self.decoder = nn.Linear(hidden_size, 1, bias=False)
    # init
    for p in self.decoder.parameters():
      init.uniform_(p, -self.hparams.critic_init_range, self.hparams.critic_init_range)
    for p in self.w.parameters():
      init.uniform_(p, -self.hparams.critic_init_range, self.hparams.critic_init_range)
      #init.uniform_(p, -0.1, 0.1)
    for p in self.w2.parameters():
      init.uniform_(p, -self.hparams.critic_init_range, self.hparams.critic_init_range)
      #init.uniform_(p, -0.1, 0.1)
      
    if self.hparams.cuda:
      self.w = self.w.cuda()
      self.w2 = self.w2.cuda()
      self.decoder = self.decoder.cuda()

  def forward(self, feature):
    #(model_feature, language_feature, data_feature) = feature
    feature, existed_src = feature
    batch_size = feature.size(0)

    if self.hparams.norm_feature:
      enc = self.w(feature / feature.sum(dim=-1).view(batch_size, -1))
    else:
      enc = self.w(feature)
    enc = torch.relu(enc)
    enc = self.w2(enc)
    enc = torch.relu(enc)
    logit = self.decoder(enc)
    return logit

