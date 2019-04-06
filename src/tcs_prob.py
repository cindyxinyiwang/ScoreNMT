import numpy as np
from scipy.special import logsumexp

def get_lan_order(base_lan, lan_dist_file="../multv-nmt/ted-train-vocab.mtok.sim-ngram.graph"):
  dists = {}
  with open(lan_dist_file, "r") as myfile:
    for line in myfile:
      base, ref, dist = line.split()
      dist = int(dist)
      if base == base_lan:
        dists[ref] = dist
  ordered_lans = sorted(dists.items(), key=lambda kv:kv[1])
  print(ordered_lans)
  #exit(0)
  return ordered_lans, dists

def prob_by_rank():
  t = 10
  base_lan = "glg"
  lan_file = "langs.txt"
  src = "data/all/ted-train.mtok.spm8000.all"
  trg = "data/all/ted-train.mtok.spm8000.eng"
  out = "data/all/glg_tcs.prob"
  lan_lists = []
  with open(lan_file) as myfile:
    for line in myfile:
      lan_lists.append(line.strip())
  el = False
  lan_order, dists = get_lan_order(base_lan, lan_dist_file="../multv-nmt/ted-train-vocab.mtok.sim-ngram.graph")
  
  # ngrams
  sim_rank = [dists[l]/100 for l in lan_lists]
  # spm8000
  #sim_rank = [kv[1]/10 for kv in lan_order]
  print(lan_lists)
  print(sim_rank)
  sim_rank = [i/t for i in sim_rank]

  src_file = open(src, 'r')
  trg_file = open(trg, 'r')
  out_file = open(out, 'w')
  lan_id = 0
  for trg_idx, trg_line in enumerate(trg_file):
    src_prob = []
    src_line, i = None, 0
    while True:
      s = src_file.readline().strip()
      i += 1
      if s == "EOF":
        lan_id = 0
        break
      if s:
        src_prob.append(sim_rank[lan_id])
      else:
        src_prob.append(0)
      lan_id += 1
    probs = []
    src_prob = src_prob
    sum_score = logsumexp(src_prob)
    for s in src_prob:
      probs.append(np.exp(s - sum_score)) 
    probs = [repr(p) for p in probs]
    out_file.write(" ".join(probs) + "\n")
   
if __name__ == "__main__":
  prob_by_rank()
