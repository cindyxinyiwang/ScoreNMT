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

from collections import defaultdict, deque
import copy
import time

TINY=1e-10

def kl_divergence(p, q):
  ret = 0
  if type(p) == list:
   for k in range(len(p)):
     ret += (-p[k] * (np.log(p[k]+TINY) - np.log(q[k]+TINY)))
  else:
   for k in p.keys():
     ret += (-p[k] * (np.log(p[k]+TINY) - np.log(q[k]+TINY)))
  return ret

def js_kl_divergence(p, q):
  ret = 0
  if type(p) == list:
    for k in range(len(p)): 
      m = (p[k] + q[k]) / 2
      ret += (-p[k] * (np.log(p[k]+TINY) - np.log(m+TINY)))
      ret += (-q[k] * (np.log(q[k]+TINY) - np.log(m+TINY)))
  else:
    for k in p.keys():
      m = (p[k] + q[k]) / 2
      ret += (-p[k] * (np.log(p[k]+TINY) - np.log(m+TINY)))
      ret += (-q[k] * (np.log(q[k]+TINY) - np.log(m+TINY)))
  return ret * 0.5

def renyi_divergence(p, q):
  ret = 0
  alpha = 0.99
  if type(p) == list:
    for k in range(len(p)):
      ret += pow(p[k], alpha) / (pow(q[k]+TINY, alpha-1)+TINY)
  else:
    for k in p.keys():
      ret += pow(p[k], alpha) / pow(q[k], alpha-1)
  return np.log(ret+TINY) / (alpha - 1)

def bha_distance(p, q):
  ret = 0
  if type(p) == list:
    for k in range(len(p)):
      ret += pow(p[k]*q[k], 0.5)
  else:
    for k in p.keys():
      ret += pow(p[k]*q[k], 0.5)
  return np.log(ret+TINY)

def cosine_similarity(p, q):
  ret = 0
  p_n, q_n = 0, 0
  if type(p) == list:
    for k in range(len(p)): 
      ret += (p[k] * q[k])
      p_n += pow(p[k], 2)
      q_n += pow(q[k], 2)
  else:
    for k in p.keys():
      ret += (p[k] * q[k])
      p_n += pow(p[k], 2)
      q_n += pow(q[k], 2)
  return ret / pow(p_n*q_n, 0.5)

def euc_distance(p, q):
  ret = 0
  if type(p) == list:
    for k in range(len(p)):
      ret += pow(p[k] - q[k], 2)
  else:
    for k in p.keys():
      ret += pow(p[k] - q[k], 2)
  return pow(ret, 0.5)

def var_distance(p, q):
  ret = 0
  if type(p) == list:
    for k in range(len(p)):
      ret += abs(p[k] - q[k])
  else:
    for k in p.keys():
      ret += abs(p[k] - q[k])
  return ret


# keep track of the state based on data we'ven seen
#class StateData(nn.Module):
class StateData(object):
  def __init__(self, hparams, data_loader):
    #super(Featurizer, self).__init__()
    self.hparams = hparams
    self.data_loader = data_loader
    self.reset_feature()
    self.step = 0
    # get x_static_ngram from base language
    x_base, x_char_base, y_base = data_loader.get_base_data()
    self.x_static_ngram_dict = self.get_ngram(x_base)
    self.x_lan_vecs = self.get_lan_vecs(data_loader)
    self.x_static_char_ngram_dict = self.condense_ngram_dict(x_char_base)
 
    if self.hparams.src_static_feature_only:
      self.num_feature = 3 * 7 + len(self.x_lan_vecs[0]) + 1 + (7)
    else:
      self.num_feature = 4 * 7 + len(self.x_lan_vecs[0]) + 1 + (7+7)

  def get_data_features(self, x_train_list, x_char_ngram_list, y_train_list, eop, lan_id, update_ngram=True):
    features = self.get_target_features(x_train_list, y_train_list, eop, update_ngram=update_ngram)
    if not self.hparams.src_static_feature_only:
      features_src = self.get_source_features(x_train_list, x_char_ngram_list, y_train_list, eop, update_ngram=update_ngram)
      features.extend(features_src)
    features_src_static = self.get_source_static_features(x_train_list, x_char_ngram_list, y_train_list, eop, lan_id, update_ngram=update_ngram)
    features.extend(features_src_static)
    #features.extend([sent_appeared_num / batch_size, ngram_appeared_num / ngram_count, weighted_ngram_appeared_num / ngram_count])
    return [features]

  def condense_ngram_dict(self, ngram_dict_list):
    final_dict = defaultdict(int)
    for x_char in ngram_dict_list:
      for ngram in x_char:
        if type(ngram) == list:
          for n in ngram:
            self.update_ngram(n, final_dict)
        else:
          self.update_ngram(ngram, final_dict)
    return final_dict

  def get_lan_vecs(self, data_loader):
    vecs = [None for _ in range(len(data_loader.lan_i2w))]
    vec_file = "lang2vecs/all58.vec"
    lines = open(vec_file, 'r').readlines()
    for i, line in enumerate(lines):
      toks = line.split()
      if toks[0] not in data_loader.lan_w2i: continue
      lan_idx = data_loader.lan_w2i[toks[0]]
      vec = []
      for t in toks[1:]:
        try:
          vec.append(float(t))
        except ValueError:
          vec.append(0)
      #print(len(vec))
      vecs[lan_idx] = vec
    return vecs

  def get_ngram(self, data_list):
    ngram_dict = defaultdict(int)
    for data in data_list:
      for n in range(4):
        for i in range(len(data)-n):
          ngram = hash(tuple(data[i:i+n]))
          ngram_dict[ngram] += 1
    return ngram_dict
 
  def get_ngram_prob(self, ngram_dict, p_ngram_dict):
    total = sum(list(ngram_dict.values()))
    p_total = sum(list(p_ngram_dict.values()))
    p, q = {}, {}
    for gram, c in ngram_dict.items():
      q[gram] = c / total
      p[gram] = p_ngram_dict[gram] / p_total
    return p, q

  def update_ngram(self, add_ngram_dict, ngram_dict):
    for gram, c in add_ngram_dict.items():
      ngram_dict[gram] += c

  def get_source_features(self, x_train_list, x_char_ngram_list, y_train_list, eop, update_ngram=True):
    features = []
    # calc ngram
    x_ngram_dict = self.get_ngram(x_train_list)
    if len(self.x_ngram_dict) == 0:
      self.update_ngram(x_ngram_dict, self.x_ngram_dict)
      update_ngram = False
    # statistics
    p, q = self.get_ngram_prob(x_ngram_dict, self.x_ngram_dict)
    js_kl = js_kl_divergence(p, q)
    features.append(js_kl)
    kl = kl_divergence(p, q)
    features.append(kl)
    renyi = renyi_divergence(p, q)
    features.append(renyi)
    bha_dist = bha_distance(p, q)
    features.append(bha_dist)
    cos_sim = cosine_similarity(p, q)
    features.append(cos_sim)
    euc_dist = euc_distance(p, q)
    features.append(euc_dist)
    var_dist = var_distance(p, q)
    features.append(var_dist)

    # char level 
    x_char_ngram_dict = self.condense_ngram_dict(x_char_ngram_list)
    if len(self.x_char_ngram_dict) == 0:
      self.update_ngram(x_char_ngram_dict, self.x_char_ngram_dict)
      update_ngram = False
    # statistics
    p, q = self.get_ngram_prob(x_char_ngram_dict, self.x_char_ngram_dict)
    js_kl = js_kl_divergence(p, q)
    features.append(js_kl)
    kl = kl_divergence(p, q)
    features.append(kl)
    renyi = renyi_divergence(p, q)
    features.append(renyi)
    bha_dist = bha_distance(p, q)
    features.append(bha_dist)
    cos_sim = cosine_similarity(p, q)
    features.append(cos_sim)
    euc_dist = euc_distance(p, q)
    features.append(euc_dist)
    var_dist = var_distance(p, q)
    features.append(var_dist)

    # update ngram
    if update_ngram:
      self.update_ngram(x_ngram_dict, self.x_ngram_dict)
      self.update_ngram(x_char_ngram_dict, self.x_char_ngram_dict)
    if eop:
      self.reset_feature()
    return features

  def get_target_features(self, x_train_list, y_train_list, eop, update_ngram=True):
    # calc ngram
    y_ngram_dict = self.get_ngram(y_train_list)
    if len(self.y_ngram_dict) == 0:
      self.update_ngram(y_ngram_dict, self.y_ngram_dict)
      update_ngram = False
    # statistics
    features = []
    p, q = self.get_ngram_prob(y_ngram_dict, self.y_ngram_dict)
    js_kl = js_kl_divergence(p, q)
    features.append(js_kl)
    kl = kl_divergence(p, q)
    features.append(kl)
    renyi = renyi_divergence(p, q)
    features.append(renyi)
    bha_dist = bha_distance(p, q)
    features.append(bha_dist)
    cos_sim = cosine_similarity(p, q)
    features.append(cos_sim)
    euc_dist = euc_distance(p, q)
    features.append(euc_dist)
    var_dist = var_distance(p, q)
    features.append(var_dist)
    # update ngram
    if update_ngram:
      self.update_ngram(y_ngram_dict, self.y_ngram_dict)
    if eop:
      self.reset_feature()
    return features

  def get_source_static_features(self, x_train_list, x_char_ngram_list, y_train_list, eop, lan_id, update_ngram=True):
    features = []
    ## calc ngram
    x_ngram_dict = self.get_ngram(x_train_list)
    # statistics
    p, q = self.get_ngram_prob(x_ngram_dict, self.x_static_ngram_dict)
    js_kl = js_kl_divergence(p, q)
    features.append(js_kl)
    kl = kl_divergence(p, q)
    features.append(kl)
    renyi = renyi_divergence(p, q)
    features.append(renyi)
    bha_dist = bha_distance(p, q)
    features.append(bha_dist)
    cos_sim = cosine_similarity(p, q)
    features.append(cos_sim)
    euc_dist = euc_distance(p, q)
    features.append(euc_dist)
    var_dist = var_distance(p, q)
    features.append(var_dist)

    # char level ngram 
    x_ngram_dict = self.condense_ngram_dict(x_char_ngram_list)
    # statistics
    p, q = self.get_ngram_prob(x_ngram_dict, self.x_static_char_ngram_dict)
    js_kl = js_kl_divergence(p, q)
    features.append(js_kl)
    kl = kl_divergence(p, q)
    features.append(kl)
    renyi = renyi_divergence(p, q)
    features.append(renyi)
    bha_dist = bha_distance(p, q)
    features.append(bha_dist)
    cos_sim = cosine_similarity(p, q)
    features.append(cos_sim)
    euc_dist = euc_distance(p, q)
    features.append(euc_dist)
    var_dist = var_distance(p, q)
    features.append(var_dist)

    # base lan feature
    vec = self.x_lan_vecs[lan_id]
    features.extend(vec)

    base_lan_id = self.data_loader.lan_w2i[self.hparams.base_lan]
    base_lan_vec = self.x_lan_vecs[base_lan_id]

    js_kl = js_kl_divergence(vec, base_lan_vec)
    features.append(js_kl)
    kl = kl_divergence(vec, base_lan_vec)
    features.append(kl)
    renyi = renyi_divergence(vec, base_lan_vec)
    features.append(renyi)
    bha_dist = bha_distance(vec, base_lan_vec)
    features.append(bha_dist)
    cos_sim = cosine_similarity(vec, base_lan_vec)
    features.append(cos_sim)
    euc_dist = euc_distance(vec, base_lan_vec)
    features.append(euc_dist)
    var_dist = var_distance(vec, base_lan_vec)
    features.append(var_dist)


    lan_dist = self.data_loader.query_lan_dist(base_lan_id, lan_id)
    features.append(lan_dist)
 
    return features

  def reset_feature(self):
    self.y_ngram_dict = defaultdict(int)
    self.x_ngram_dict = defaultdict(int)
    self.x_char_ngram_dict = defaultdict(int)
    self.y_weighted_ngram_dict = defaultdict(int)
    self.x_weighted_ngram_dict = defaultdict(int)
    self.step = 0


