import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np
from utils import *

class MlpAttn(nn.Module):
  def __init__(self, hparams):
    super(MlpAttn, self).__init__()
    self.hparams = hparams
    self.dropout = nn.Dropout(hparams.dropout)
    self.w_trg = nn.Linear(self.hparams.d_model, self.hparams.d_model)
    self.w_att = nn.Linear(self.hparams.d_model, 1)
    if self.hparams.cuda:
      self.w_trg = self.w_trg.cuda()
      self.w_att = self.w_att.cuda()
  
  def forward(self, q, k, v, attn_mask=None):
    batch_size, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()
    # v is bi-directional encoding of source
    assert d_k == d_q 
    #assert 2*d_k == d_v
    assert len_k == len_v
    # (batch_size, len_k, d_k)
    att_src_hidden = nn.functional.tanh(k + self.w_trg(q).unsqueeze(1))
    # (batch_size, len_k)
    att_src_weights = self.w_att(att_src_hidden).squeeze(2)
    if not attn_mask is None:
      att_src_weights.data.masked_fill_(attn_mask, -self.hparams.inf)
    att_src_weights = F.softmax(att_src_weights, dim=-1)
    att_src_weights = self.dropout(att_src_weights)
    ctx = torch.bmm(att_src_weights.unsqueeze(1), v).squeeze(1)
    return ctx


class LayerNormalization(nn.Module):
  def __init__(self, d_hid, eps=1e-9):
    super(LayerNormalization, self).__init__()

    self.d_hid = d_hid
    self.eps = eps
    self.scale = nn.Parameter(torch.ones(self.d_hid), requires_grad=True)
    self.offset= nn.Parameter(torch.zeros(self.d_hid), requires_grad=True)

  def forward(self, x):
    assert x.dim() >= 2
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return self.scale * (x - mean) / (std + self.eps) + self.offset

class DotProdAttn(nn.Module):
  def __init__(self, hparams):
    super(DotProdAttn, self).__init__()
    self.dropout = nn.Dropout(hparams.dropout)
    #self.src_enc_linear = nn.Linear(hparams.d_model * 2, hparams.d_model)
    self.softmax = nn.Softmax(dim=-1)
    self.hparams = hparams
    self.temp = np.power(hparams.d_model, 0.5)

  def forward(self, q, k, v, attn_mask = None):
    """ 
    dot prodct attention: (q * k.T) * v
    Args:
      q: [batch_size, d_q] (target state)
      k: [batch_size, len_k, d_k] (source enc key vectors)
      v: [batch_size, len_v, d_v] (source encoding vectors)
      attn_mask: [batch_size, len_k] (source mask)
    Return:
      attn: [batch_size, d_v]
    """
    batch_size, d_q = q.size()
    batch_size, len_k, d_k = k.size()
    batch_size, len_v, d_v = v.size()
    # v is bi-directional encoding of source
    assert d_k == d_q 
    #assert 2*d_k == d_v
    assert len_k == len_v
    # [batch_size, len_k, d_model]
    #k_vec = self.src_enc_linear(k)
    # [batch_size, len_k]
    attn_weight = torch.bmm(k, q.unsqueeze(2)).squeeze(2) / self.temp
    if not attn_mask is None:
      attn_weight.data.masked_fill_(attn_mask, -self.hparams.inf)
    attn_weight = self.softmax(attn_weight)
    attn_weight = self.dropout(attn_weight)
    # [batch_size, d_v]
    ctx = torch.bmm(attn_weight.unsqueeze(1), v).squeeze(1)
    return ctx

class MultiHeadAttn(nn.Module):
  def __init__(self, hparams):
    super(MultiHeadAttn, self).__init__()

    self.hparams = hparams

    self.attention = DotProdAttn(hparams)
    self.layer_norm = LayerNormalization(hparams.d_model)

    # projection of concatenated attn
    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v

    Q, K, V = [], [], []
    for head_id in range(n_heads):
      q = nn.Linear(d_model, d_q, bias=False)
      k = nn.Linear(d_model, d_k, bias=False)
      v = nn.Linear(d_model, d_v, bias=False)
      init_param(q.weight, init_type="uniform", init_range=hparams.init_range)
      init_param(k.weight, init_type="uniform", init_range=hparams.init_range)
      init_param(v.weight, init_type="uniform", init_range=hparams.init_range)
      Q.append(q)
      K.append(k)
      V.append(v)
    self.Q = nn.ModuleList(Q)
    self.K = nn.ModuleList(K)
    self.V = nn.ModuleList(V)
    if self.hparams.cuda:
      self.Q = self.Q.cuda()
      self.K = self.K.cuda()
      self.V = self.V.cuda()

    self.w_proj = nn.Linear(n_heads * d_v, d_model, bias=False)
    init_param(self.w_proj.weight, init_type="uniform", init_range=hparams.init_range)
    if self.hparams.cuda:
      self.w_proj = self.w_proj.cuda()

  def forward(self, q, k, v, attn_mask=None):
    """Performs the following computations:
         head[i] = Attention(q * w_q[i], k * w_k[i], v * w_v[i])
         outputs = concat(all head[i]) * self.w_proj
    Args:
      q: [batch_size, len_q, d_q].
      k: [batch_size, len_k, d_k].
      v: [batch_size, len_v, d_v].
    Must have: len_k == len_v
    Note: This batch_size is in general NOT the training batch_size, as
      both sentences and time steps are batched together for efficiency.
    Returns:
      outputs: [batch_size, len_q, d_model].
    """

    residual = q 

    n_heads = self.hparams.n_heads
    d_model = self.hparams.d_model
    d_q = self.hparams.d_k
    d_k = self.hparams.d_k
    d_v = self.hparams.d_v
    batch_size = q.size(0)

    heads = []
    for Q, K, V in zip(self.Q, self.K, self.V):
      head_q, head_k, head_v = Q(q), K(k), V(v)
      head = self.attention(head_q, head_k, head_v, attn_mask=attn_mask)
      heads.append(head)

    outputs = torch.cat(heads, dim=-1).contiguous().view(batch_size, n_heads * d_v)
    outputs = self.w_proj(outputs)
    if not hasattr(self.hparams, "residue") or self.hparams.residue == 1:
      outputs = outputs + residual
    if not hasattr(self.hparams, "layer_norm") or self.hparams.layer_norm == 1: 
      outputs = self.layer_norm(outputs)

    return outputs

class Encoder(nn.Module):
  def __init__(self, hparams, *args, **kwargs):
    super(Encoder, self).__init__()

    self.hparams = hparams
    #print("d_word_vec", self.hparams.d_word_vec)
    if not self.hparams.src_char_only:
      self.word_emb = nn.Embedding(self.hparams.src_vocab_size,
                                   self.hparams.d_word_vec,
                                   padding_idx=hparams.pad_id)

    if self.hparams.char_ngram_n > 0:
      self.char_emb_proj = nn.Linear(self.hparams.src_char_vsize, self.hparams.d_word_vec, bias=False)
      if self.hparams.cuda:
        self.char_emb_proj = self.char_emb_proj.cuda()
    elif self.hparams.char_input:
      self.char_emb = nn.Embedding(self.hparams.src_char_vsize, self.hparams.d_word_vec, padding_idx=hparams.pad_id)
      if self.hparams.cuda:
        self.char_emb = self.char_emb.cuda()

    if self.hparams.char_comb == "add":
      d_word_vec = self.hparams.d_word_vec
    elif self.hparams.char_comb == "cat":
      d_word_vec = self.hparams.d_word_vec * 2

    self.layer = nn.LSTM(d_word_vec, 
                         self.hparams.d_model, 
                         bidirectional=True, 
                         dropout=hparams.dropout)

    # bridge from encoder state to decoder init state
    self.bridge = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    
    self.dropout = nn.Dropout(self.hparams.dropout)
    if self.hparams.cuda:
      if not self.hparams.src_char_only:
        self.word_emb = self.word_emb.cuda()
      self.layer = self.layer.cuda()
      self.dropout = self.dropout.cuda()
      self.bridge = self.bridge.cuda()

  def forward(self, x_train, x_len, x_train_char=None):
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
    if self.hparams.src_char_only:
      word_emb = Variable(torch.zeros(max_len, batch_size, self.hparams.d_word_vec), volatile=True)
      if self.hparams.cuda: word_emb = word_emb.cuda()
    else:
      word_emb = self.word_emb(x_train)
      word_emb = self.dropout(word_emb)
    if self.hparams.char_ngram_n > 0:
      for idx, x_char_sent in enumerate(x_train_char):
        emb = Variable(x_char_sent.to_dense(), requires_grad=False)
        if self.hparams.cuda: emb = emb.cuda()
        x_char_sent = torch.tanh(self.char_emb_proj(emb))
        x_train_char[idx] = x_char_sent
      char_emb = torch.stack(x_train_char, dim=0).permute(1, 0, 2)
      if self.hparams.char_comb == 'add':
        if not self.hparams.char_temp:
          word_emb = word_emb + char_emb
        elif self.hparams.char_temp < 1:
          word_emb = word_emb * (1-self.hparams.char_temp) + char_emb * self.hparams.char_temp
        elif self.hparams.char_temp > 1:
          word_emb = word_emb + char_emb * self.hparams.char_temp
      elif self.hparams.char_comb == 'cat':
        word_emb = torch.cat([word_emb, char_emb], dim=-1)
    elif self.hparams.char_input:
      # [batch_size, max_len, char_len, d_word_vec]
      char_emb = self.char_emb(x_train_char.permute(1, 0, 2))
      char_emb = char_emb.sum(dim=2)
      if self.hparams.char_comb == 'add':
        if not self.hparams.char_temp:
          word_emb = word_emb + char_emb
        elif self.hparams.char_temp < 1:
          word_emb = word_emb * (1-self.hparams.char_temp) + char_emb * self.hparams.char_temp
        elif self.hparams.char_temp > 1:
          word_emb = word_emb + char_emb * self.hparams.char_temp
      elif self.hparams.char_comb == 'cat':
        word_emb = torch.cat([word_emb, char_emb], dim=-1)

    packed_word_emb = pack_padded_sequence(word_emb, x_len)
    enc_output, (ht, ct) = self.layer(packed_word_emb)
    enc_output, _ = pad_packed_sequence(enc_output,  padding_value=self.hparams.pad_id)
    enc_output = enc_output.permute(1, 0, 2)

    dec_init_cell = self.bridge(torch.cat([ct[0], ct[1]], 1))
    dec_init_state = F.tanh(dec_init_cell)
    dec_init = (dec_init_state, dec_init_cell)

    #dec_init_cell = torch.cat([ct[0], ct[1]], 1)
    #dec_init_state = F.tanh(dec_init_cell)
    #dec_init = (dec_init_state, dec_init_cell)
    return enc_output, dec_init

class Decoder(nn.Module):
  def __init__(self, hparams):
    super(Decoder, self).__init__()
    self.hparams = hparams
    
    #self.attention = DotProdAttn(hparams)
    self.attention = MlpAttn(hparams)
    # transform [ctx, h_t] to readout state vectors before softmax
    self.ctx_to_readout = nn.Linear(hparams.d_model * 2 + hparams.d_model, hparams.d_model, bias=False)
    #self.ctx_to_readout = nn.Linear(hparams.d_model + hparams.d_model, hparams.d_model, bias=False)
    self.readout = nn.Linear(hparams.d_model, hparams.trg_vocab_size, bias=False)
    self.word_emb = nn.Embedding(self.hparams.trg_vocab_size,
                                 self.hparams.d_word_vec,
                                 padding_idx=hparams.pad_id)

    if self.hparams.char_ngram_n > 0:
      self.char_emb_proj = nn.Linear(self.hparams.trg_char_vsize, self.hparams.d_word_vec, bias=False)
      if self.hparams.cuda:
        self.char_emb_proj = self.char_emb_proj.cuda()
    elif self.hparams.char_input:
      self.char_emb = nn.Embedding(self.hparams.trg_char_vsize, self.hparams.d_word_vec, padding_idx=hparams.pad_id)
      if self.hparams.cuda:
        self.char_emb = self.char_emb.cuda()

    if self.hparams.char_comb == "add":
      d_word_vec = self.hparams.d_word_vec
    elif self.hparams.char_comb == "cat":
      d_word_vec = self.hparams.d_word_vec * 2

    # input: [y_t-1, input_feed]
    #self.layer = nn.LSTMCell(d_word_vec + hparams.d_model, 
    self.layer = nn.LSTMCell(d_word_vec + hparams.d_model * 2, 
                             hparams.d_model)
    self.dropout = nn.Dropout(hparams.dropout)
    if self.hparams.cuda:
      self.ctx_to_readout = self.ctx_to_readout.cuda()
      self.readout = self.readout.cuda()
      self.word_emb = self.word_emb.cuda()
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
    #input_feed = Variable(torch.zeros(batch_size, self.hparams.d_model), requires_grad=False)
    if self.hparams.cuda:
      input_feed = input_feed.cuda()
    # [batch_size, y_len, d_word_vec]
    trg_emb = self.word_emb(y_train)
    if self.hparams.char_ngram_n > 0:
      for idx, y_char_sent in enumerate(y_train_char):
        emb = Variable(y_char_sent.to_dense(), requires_grad=False)[:-1,:]
        if self.hparams.cuda: emb = emb.cuda()
        y_char_sent = torch.tanh(self.char_emb_proj(emb))
        y_train_char[idx] = y_char_sent
      char_emb = torch.stack(y_train_char, dim=0)
      if self.hparams.char_comb == 'add':
        if not self.hparams.char_temp:
          trg_emb = trg_emb + char_emb
        elif self.hparams.char_temp < 1:
          trg_emb = trg_emb * (1-self.hparams.char_temp) + char_emb * self.hparams.char_temp
        elif self.hparams.char_temp > 1:
          trg_emb = trg_emb + char_emb * self.hparams.char_temp
      elif self.hparams.char_comb == 'cat':
        trg_emb = torch.cat([trg_emb, char_emb], dim=-1)
    elif self.hparams.char_input:
      # [batch_size, max_len, char_len, d_word_vec]
      char_emb = self.char_emb(y_train_char)
      char_emb = char_emb.sum(dim=2)[:,:-1,:]
      if self.hparams.char_comb == 'add':
        if not self.hparams.char_temp:
          trg_emb = trg_emb + char_emb
        elif self.hparams.char_temp < 1:
          trg_emb = trg_emb * (1-self.hparams.char_temp) + char_emb * self.hparams.char_temp
        elif self.hparams_char_temp > 1:
          trg_emb = trg_emb + char_emb * self.hparams.char_temp
      elif self.hparams.char_comb == 'cat':
        trg_emb = torch.cat([trg_emb, char_emb], dim=-1)

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
    # [len_y, batch_size, trg_vocab_size]
    logits = self.readout(torch.stack(pre_readouts)).transpose(0, 1).contiguous()
    return logits

  def step(self, x_enc, x_enc_k, y_tm1, dec_state, ctx_t, data):
    y_emb_tm1 = self.word_emb(y_tm1)
    if self.hparams.char_ngram_n > 0:
      emb = data.get_char_emb(y_tm1.item())
      if self.hparams.cuda: emb = emb.cuda()
      emb = torch.tanh(self.char_emb_proj(emb))
      if self.hparams.char_comb == 'add':
        if not self.hparams.char_temp:
          y_emb_tm1 = y_emb_tm1 + emb
        elif self.hparams.char_temp < 1:
          y_emb_tm1 = y_emb_tm1 * (1 - self.hparams.char_temp) + emb * self.hparams.char_temp
        elif self.hparams.char_temp > 1:
          y_emb_tm1 = y_emb_tm1 + emb * self.hparams.char_temp
      elif self.hparams.char_comb == 'cat':
        y_emb_tm1 = torch.cat([y_emb_tm1, emb], dim=-1)
    elif self.hparams.char_input:
      # [1, char_len]
      char = data.get_char_emb(y_tm1.item())     
      emb = self.char_emb(char).sum(dim=1)
      if self.hparams.char_comb == 'add':
        if self.hparams.char_temp < 1:
          y_emb_tm1 = y_emb_tm1 * (1 - self.hparams.char_temp) + emb * self.hparams.char_temp
        elif self.hparams.char_temp > 1:
          y_emb_tm1 = y_emb_tm1 + emb * self.hparams.char_temp
        else:
          y_emb_tm1 = y_emb_tm1 + emb
      elif self.hparams.char_comb == 'cat':
        y_emb_tm1 = torch.cat([y_emb_tm1, emb], dim=-1)

    y_input = torch.cat([y_emb_tm1, ctx_t], dim=1)
    #print (y_input.size())
    #print (dec_state[0].size())
    #print (dec_state[1].size())
    h_t, c_t = self.layer(y_input, dec_state)
    ctx = self.attention(h_t, x_enc_k, x_enc)
    pre_readout = F.tanh(self.ctx_to_readout(torch.cat([h_t, ctx], dim=1)))
    logits = self.readout(pre_readout)

    return logits, (h_t, c_t), ctx

class Seq2Seq(nn.Module):
  
  def __init__(self, hparams, data):
    super(Seq2Seq, self).__init__()
    self.encoder = Encoder(hparams)
    self.decoder = Decoder(hparams)
    self.data = data
    # transform encoder state vectors into attention key vector
    self.enc_to_k = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    self.hparams = hparams
    if self.hparams.cuda:
      self.enc_to_k = self.enc_to_k.cuda()

  def forward(self, x_train, x_mask, x_len, y_train, y_mask, y_len, x_train_char_sparse=None, y_train_char_sparse=None):
    # [batch_size, x_len, d_model * 2]
    x_enc, dec_init = self.encoder(x_train, x_len, x_train_char_sparse)
    x_enc_k = self.enc_to_k(x_enc)
    #x_enc_k = x_enc
    # [batch_size, y_len-1, trg_vocab_size]
    logits = self.decoder(x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, y_train_char_sparse)
    return logits

  def translate(self, x_train, max_len=100, beam_size=5, poly_norm_m=0, x_train_char=None, y_train_char=None):
    hyps = []
    for i, x in enumerate(x_train):
      if x_train_char:
        x_char = x_train_char[i]
      else:
        x_char = None
      x = Variable(torch.LongTensor(x), volatile=True)
      if self.hparams.cuda:
        x = x.cuda()
      hyp = self.translate_sent(x, max_len=max_len, beam_size=beam_size, poly_norm_m=poly_norm_m, x_train_char=x_char)[0]
      hyps.append(hyp.y[1:-1])
    return hyps

  def translate_sent(self, x_train, max_len=100, beam_size=5, poly_norm_m=0, x_train_char=None):
    assert len(x_train.size()) == 1
    x_len = [x_train.size(0)]
    x_train = x_train.unsqueeze(0)
    x_enc, dec_init = self.encoder(x_train, x_len, x_train_char)
    x_enc_k = self.enc_to_k(x_enc)
    #x_enc_k = x_enc
    length = 0
    completed_hyp = []
    input_feed = Variable(torch.zeros(1, self.hparams.d_model * 2), requires_grad=False)
    #input_feed = Variable(torch.zeros(1, self.hparams.d_model), requires_grad=False)
    if self.hparams.cuda:
      input_feed = input_feed.cuda()
    active_hyp = [Hyp(state=dec_init, y=[self.hparams.bos_id], ctx_tm1=input_feed, score=0.)]
    while len(completed_hyp) < beam_size and length < max_len:
      length += 1
      new_hyp_score_list = []
      for i, hyp in enumerate(active_hyp):
        y_tm1 = Variable(torch.LongTensor([int(hyp.y[-1])] ), volatile=True)
        if self.hparams.cuda:
          y_tm1 = y_tm1.cuda()
        logits, dec_state, ctx = self.decoder.step(x_enc, x_enc_k, y_tm1, hyp.state, hyp.ctx_tm1, self.data)
        hyp.state = dec_state
        hyp.ctx_tm1 = ctx 

        p_t = F.log_softmax(logits, -1).data
        if poly_norm_m > 0 and length > 1:
          new_hyp_scores = (hyp.score * pow(length-1, poly_norm_m) + p_t) / pow(length, poly_norm_m)
        else:
          new_hyp_scores = hyp.score + p_t 
        new_hyp_score_list.append(new_hyp_scores)
      live_hyp_num = beam_size - len(completed_hyp)
      new_hyp_scores = np.concatenate(new_hyp_score_list).flatten()
      new_hyp_pos = (-new_hyp_scores).argsort()[:live_hyp_num]
      prev_hyp_ids = new_hyp_pos / self.hparams.trg_vocab_size
      word_ids = new_hyp_pos % self.hparams.trg_vocab_size
      new_hyp_scores = new_hyp_scores[new_hyp_pos]

      new_hypotheses = []
      for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
        prev_hyp = active_hyp[int(prev_hyp_id)]
        hyp = Hyp(state=prev_hyp.state, y=prev_hyp.y+[word_id], ctx_tm1=prev_hyp.ctx_tm1, score=hyp_score)
        if word_id == self.hparams.eos_id:
          completed_hyp.append(hyp)
        else:
          new_hypotheses.append(hyp)
        #print(word_id, hyp_score)
      #exit(0)
      active_hyp = new_hypotheses

    if len(completed_hyp) == 0:
      completed_hyp.append(active_hyp[0])
    return sorted(completed_hyp, key=lambda x: x.score, reverse=True)

class Hyp(object):
  def __init__(self, state, y, ctx_tm1, score):
    self.state = state
    self.y = y 
    self.ctx_tm1 = ctx_tm1
    self.score = score
