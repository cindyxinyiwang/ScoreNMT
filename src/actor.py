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

from collections import defaultdict, deque
import copy
import time

class Actor(nn.Module):
  def __init__(self, hparams, num_feature, lan_dist_vec):
    super(Actor, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden
    self.lan_dist_vec = Variable(torch.FloatTensor(lan_dist_vec.tolist()) / 10)
    self.w = nn.Linear(num_feature, hidden_size, bias=False)
    self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
    self.decoder = nn.Linear(hidden_size, self.hparams.lan_size, bias=False)
    # init
    for p in self.decoder.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
    for p in self.w.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
      #init.uniform_(p, -0.1, 0.1)
    for p in self.w2.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
      #init.uniform_(p, -0.1, 0.1)
      
    if self.hparams.add_bias:
      self.bias = nn.Linear(1, self.hparams.lan_size, bias=False)
      self.bias.weight = torch.nn.Parameter(torch.FloatTensor([[self.hparams.bias for _ in range(self.hparams.lan_size)]]))
      self.bias.weight.requires_grad = False
    #self.decoder.bias = torch.nn.Parameter(torch.FloatTensor(lan_dist_vec))
    if self.hparams.cuda:
      self.lan_dist_vec = self.lan_dist_vec.cuda()
      self.w = self.w.cuda()
      self.w2 = self.w2.cuda()
      self.decoder = self.decoder.cuda()
      if self.hparams.add_bias:
        self.bias = self.bias.cuda()

  def forward(self, feature):
    #(model_feature, language_feature, data_feature) = feature
    feature, existed_src = feature
    batch_size = feature.size(0)

    if self.hparams.norm_feature:
      enc = self.w(feature / feature.sum(dim=-1))
    else:
      enc = self.w(feature)
    enc = torch.relu(enc)
    enc = self.w2(enc)
    enc = torch.relu(enc)
    if self.hparams.add_bias:
      bias = self.bias.weight * existed_src * self.lan_dist_vec
      logit = self.decoder(enc) + bias
    else:
      logit = self.decoder(enc)
    return logit

class HeuristicActor(nn.Module):
  def __init__(self, hparams, num_feature, lan_dist_vec):
    super(HeuristicActor, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden
    #self.lan_dist_vec = Variable(torch.FloatTensor(lan_dist_vec.tolist()) / 10)
    self.lan_dist_vec = Variable(torch.FloatTensor(lan_dist_vec.tolist()))
      
    self.bias = nn.Linear(1, self.hparams.lan_size, bias=False)
    self.bias.weight = torch.nn.Parameter(torch.FloatTensor([[self.hparams.bias for _ in range(self.hparams.lan_size)]]))
    self.bias.weight.requires_grad = False
    if self.hparams.cuda:
      self.bias = self.bias.cuda()
      self.lan_dist_vec = self.lan_dist_vec.cuda()

  def forward(self, feature):
    #(model_feature, language_feature, data_feature) = feature
    feature, existed_src = feature

    bias_logit = self.bias.weight * existed_src * self.lan_dist_vec
    return bias_logit

class InitActor(nn.Module):
  def __init__(self, hparams, num_feature, lan_dist_vec):
    super(InitActor, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden
    lan_vector = [0 for _ in range(self.hparams.lan_size)]
    lan_vector[self.hparams.base_lan_id] = 100
    self.lan_dist_vec = Variable(torch.FloatTensor(lan_vector))
    if self.hparams.cuda:
      self.lan_dist_vec = self.lan_dist_vec.cuda()

  def forward(self, feature):
    #(model_feature, language_feature, data_feature) = feature
    feature, existed_src = feature

    logit = existed_src * self.lan_dist_vec
    return logit

class EmbActor(nn.Module):
  def __init__(self, hparams, lan_dist_vec):
    super(EmbActor, self).__init__()
    self.hparams = hparams
    hidden_size = hparams.d_hidden
    self.lan_dist_vec = Variable(torch.FloatTensor(lan_dist_vec.tolist()) / 10)
    self.w_src = nn.Linear(hparams.d_word_vec, hidden_size, bias=False)
    self.w_trg = nn.Linear(hparams.d_word_vec, hidden_size, bias=False)
    self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
    self.decoder = nn.Linear(hidden_size, self.hparams.lan_size, bias=False)
    # init
    for p in self.decoder.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
    for p in self.w_trg.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
    for p in self.w_src.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
    for p in self.w2.parameters():
      init.uniform_(p, -self.hparams.actor_init_range, self.hparams.actor_init_range)
      
    if self.hparams.add_bias:
      self.bias = nn.Linear(1, self.hparams.lan_size, bias=False)
      self.bias.weight = torch.nn.Parameter(torch.FloatTensor([[self.hparams.bias for _ in range(self.hparams.lan_size)]]))
      self.bias.weight.requires_grad = False
    if self.hparams.cuda:
      self.lan_dist_vec = self.lan_dist_vec.cuda()
      self.w_src = self.w_src.cuda()
      self.w_trg = self.w_trg.cuda()
      self.w2 = self.w2.cuda()
      if self.hparams.add_bias:
        self.bias = self.bias.cuda()
      self.decoder = self.decoder.cuda()

  def forward(self, feature):
    ([src_emb, trg_emb], existed_src) = feature
    batch_size = src_emb.size(0)
    #batch_size, lan_size, d_word_vec
    #batch_size, 1, d_word_vec
    enc_src = self.w_src(src_emb)
    enc_trg = self.w_trg(trg_emb)
    # lan_size+1, d_model
    enc = self.w2(torch.cat([enc_src, enc_trg], dim=1)).sum(dim=1).view(batch_size,  -1)
    enc = torch.tanh(enc)
    if self.hparams.add_bias:
      bias = self.bias.weight * existed_src * self.lan_dist_vec
      logit = self.decoder(enc) + bias
    else:
      logit = self.decoder(enc)
    return logit


class ActorEncoder(nn.Module):
  def __init__(self, hparams, emb, *args, **kwargs):
    super(ActorEncoder, self).__init__()

    self.hparams = hparams
    self.word_emb = nn.Embedding(emb.weight.size(0), emb.weight.size(1))
    self.word_emb.weight.data = emb.weight.data
    self.word_emb.weight.requires_grad = False

    self.layer = nn.LSTM(self.hparams.d_word_vec, 
                         self.hparams.d_model, 
                         bidirectional=True, 
                         dropout=hparams.dropout)

    # bridge from encoder state to decoder init state
    self.bridge = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    
    self.dropout = nn.Dropout(self.hparams.dropout)
    if self.hparams.cuda:
      self.word_emb = self.word_emb.cuda()
      self.layer = self.layer.cuda()
      self.dropout = self.dropout.cuda()
      self.bridge = self.bridge.cuda()

  def forward(self, x_train, x_len, x_train_char=None, file_idx=None):
    """Performs a forward pass.
    Args:
      x_train: Torch Tensor of size [batch_size, max_len]
      x_mask: Torch Tensor of size [batch_size, max_len]. 1 means to ignore a
        position.
      x_len: [batch_size,]
    Returns:
      enc_output: Tensor of size [batch_size, max_len, d_model].
    """
    batch_size, max_len = x_train.size()
    x_train = x_train.transpose(0, 1)
    # [batch_size, max_len, d_word_vec]
    word_emb = self.word_emb(x_train)
    word_emb = self.dropout(word_emb)

    packed_word_emb = pack_padded_sequence(word_emb, x_len)
    enc_output, (ht, ct) = self.layer(packed_word_emb)
    enc_output, _ = pad_packed_sequence(enc_output,  padding_value=self.hparams.pad_id)
    #enc_output, (ht, ct) = self.layer(word_emb)
    enc_output = enc_output.permute(1, 0, 2)

    dec_init_cell = self.bridge(torch.cat([ct[0], ct[1]], 1))
    dec_init_state = F.tanh(dec_init_cell)
    dec_init = (dec_init_state, dec_init_cell)

    return enc_output, dec_init

class ActorDecoder(nn.Module):
  def __init__(self, hparams, emb):
    super(ActorDecoder, self).__init__()
    self.hparams = hparams
    
    self.attention = MlpAttn(hparams)
    # transform [ctx, h_t] to readout state vectors before softmax
    self.ctx_to_readout = nn.Linear(hparams.d_model * 2 + hparams.d_model, hparams.d_model, bias=False)
    
    self.word_emb = nn.Embedding(emb.weight.size(0), emb.weight.size(1))
    self.word_emb.weight.data = emb.weight.data
    self.word_emb.weight.requires_grad = False

    # input: [y_t-1, input_feed]
    self.layer = nn.LSTMCell(self.hparams.d_word_vec + hparams.d_model * 2, 
                             hparams.d_model)
    self.dropout = nn.Dropout(hparams.dropout)
    if self.hparams.cuda:
      self.ctx_to_readout = self.ctx_to_readout.cuda()
      self.layer = self.layer.cuda()
      self.dropout = self.dropout.cuda()

  def forward(self, x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, y_train_char=None):
    # get decoder init state and cell, use x_ct
    """
    x_enc: [batch_size, max_x_len, d_model * 2]
    """
    batch_size_x = x_enc.size()[0]
    batch_size, y_max_len = y_train.size()
    assert batch_size_x == batch_size
    hidden = dec_init 
    input_feed = Variable(torch.zeros(batch_size, self.hparams.d_model * 2), requires_grad=False)
    if self.hparams.cuda:
      input_feed = input_feed.cuda()
    # [batch_size, y_len, d_word_vec]
    trg_emb = self.word_emb(y_train)

    pre_readouts = []
    logits = []
    for t in range(y_max_len):
      y_emb_tm1 = trg_emb[:, t, :]
      y_input = torch.cat([y_emb_tm1, input_feed], dim=1)
      
      h_t, c_t = self.layer(y_input, hidden)
      ctx = self.attention(h_t, x_enc_k, x_enc, attn_mask=x_mask)
      pre_readout = F.tanh(self.ctx_to_readout(torch.cat([h_t, ctx], dim=1)))
      pre_readout = self.dropout(pre_readout)
      pre_readouts.append(pre_readout)

      input_feed = ctx
      hidden = (h_t, c_t)
    # [len_y, batch_size, d_model]
    prereadout = torch.stack(pre_readouts)
    return prereadout

class ActorNMT(nn.Module):
  def __init__(self, hparams, src_emb, trg_emb):
    super(ActorNMT, self).__init__()
    self.hparams = hparams
    self.actor_encoder = ActorEncoder(hparams, src_emb)
    self.actor_decoder = ActorDecoder(hparams, trg_emb)
    self.enc_to_k = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    self.readout_to_score = nn.Linear(hparams.d_model, 1, bias=False)
    if self.hparams.cuda:
      self.enc_to_k = self.enc_to_k.cuda()
      self.readout_to_score = self.readout_to_score.cuda()

  def forward(self, x_train, x_mask, x_len, x_pos_emb_idxs, y_train, y_mask, y_len, y_pos_emb_idxs):
    x_enc, dec_init = self.actor_encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    #x_enc_k = x_enc
    # [y_len-1, batch_size, d_model]
    pre_readouts = self.actor_decoder(x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask).permute(1, 0, 2)
    # [batch_size, 1]
    scores = torch.sigmoid(self.readout_to_score(pre_readouts).squeeze(2).sum(dim=-1)).unsqueeze(1)
    return scores

