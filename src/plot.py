from collections import defaultdict
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["font.size"] = 33
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["legend.fontsize"] = "large"
mpl.rcParams["legend.framealpha"] = None
mpl.rcParams["figure.titlesize"] = "medium"
mpl.rcParams["axes.labelsize"] = "large"
mpl.rcParams["text.usetex"] = True
mpl.rcParams['figure.figsize'] = 10, 7.5

#f, axarr = plt.subplots(2, 2)

lans = ["aze", "bel", "glg", "slk", "tur", "rus", "por", "ces"] 
lan_num = [5946, 4509, 10017, 61470, 182470, 208458, 184755, 103093]


def get_bucket_count():
  src = "data/tiny/ted-train.mtok.spm8000.tiny"
  out_file = "data/tiny/bucket_count.txt"
  data_bucket, src_exist = {}, []
  with open(src, 'r') as src_file:
    for s in src_file:
      s = s.strip()
      if s == "EOF":
        src_exist = tuple(src_exist)
        if not src_exist in data_bucket:
          data_bucket[src_exist] = 0
        data_bucket[src_exist] += 1
        cur_lan = 0
        src_exist = []
      else:
        toks = s.split()
        if len(toks) > 0:
          src_exist.append(1)
        else:
          src_exist.append(0)
  with open(out_file, "w") as myfile:
    for key, v in data_bucket.items():
      for k in key: myfile.write(str(k))
      myfile.write(" ")
      myfile.write(str(v))
      myfile.write("\n")

def load_bucket_count():
  out_file = "data/tiny/bucket_count.txt"
  data_bucket_count = {}
  with open(out_file, 'r') as myfile:
    for line in myfile:
      vals = line.split()
      key = tuple([int(i) for i in vals[0]])
      count = int(vals[1])
      data_bucket_count[key] = count
  return data_bucket_count

def get_probs(filename):
  lan_probs_list = []
  cur_probs = []
  lines = open(filename, 'r').readlines()
  prob_reg = re.compile("prob=\[(.*)\]")  
  for i, line in enumerate(lines):
    if line.startswith("grad="):
      lan_probs_list.append(np.array(cur_probs))
      cur_probs = []
    elif line.startswith("prob="):
      p = prob_reg.match(line).group(1)
      p = np.array([float(i) for i in p.split(", ")])
      cur_probs.append(p)
  return lan_probs_list

def prob_plot(name, stdout):
  linewidth = 5.0
  markersize = 10

  bucket_count = load_bucket_count()
  probs_list = get_probs(stdout)
  final_probs_tur = []
  final_probs_rus = []
  final_probs_por = []
  final_probs_ces = []
  for prob in probs_list:
    #ave_prob = np.sum(prob, axis=0) / np.sum(prob > 0, axis=0)
    tur_c, rus_c, por_c, ces_c = 0, 0, 0, 0
    for p in prob:
      exist = tuple(p>0)
      #if exist in bucket_count:
      #  count = np.multiply(bucket_count[exist], p)
      #  tur_c += count[4]
      #  rus_c += count[5]
      #  por_c += count[6]
      #  ces_c += count[7]
      if exist == (0,0,0,0,1,1,1,1):
        ave_prob = p
        break
    final_probs_tur.append(ave_prob[4])
    final_probs_rus.append(ave_prob[5])
    final_probs_por.append(ave_prob[6])
    final_probs_ces.append(ave_prob[7])

    #final_probs_tur.append(tur_c / lan_num[4])
    #final_probs_rus.append(rus_c / lan_num[5])
    #final_probs_por.append(por_c / lan_num[6])
    #final_probs_ces.append(ces_c / lan_num[7])

    #final_probs_tur.append(tur_c)
    #final_probs_rus.append(rus_c)
    #final_probs_por.append(por_c)
    #final_probs_ces.append(ces_c)


  #ax = axarr[i, j]
  markers = ['o', 'v', 's', '>']

  steps = [i for i in range(len(final_probs_tur))]
  plt.plot(steps, final_probs_tur, linewidth=linewidth, markersize=markersize, marker=markers[0])
  plt.plot(steps, final_probs_rus, linewidth=linewidth, markersize=markersize, marker=markers[1])
  plt.plot(steps, final_probs_por, linewidth=linewidth, markersize=markersize, marker=markers[2])
  plt.plot(steps, final_probs_ces, linewidth=linewidth, markersize=markersize, marker=markers[3])

  legends = ["tur", "rus", "por", "ces"] 
  plt.legend(legends, ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)
  #plt.xlabel("Step")
  #plt.ylabel("Dev ppl")
  #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
  plt.savefig(name + "_probs_plot.eps", format='eps', bbox_inches='tight')


if __name__ == "__main__":
  #get_bucket_count()
  prob_plot("bel_hs", "outputs_plots/hs-cosine-zero_one_tiny_bel/stdout")
  plt.clf()
  prob_plot("bel_uni", "outputs_plots/uniform-cosine-zero_one_tiny_bel/stdout")
  plt.clf()
  prob_plot("aze_hs", "outputs_plots/hs-cosine-zero_one-v2_tiny_aze/stdout")
  plt.clf()
  prob_plot("aze_uni", "outputs_plots/uniform-cosine-zero_one_tiny_aze/stdout")
  plt.clf()
  prob_plot("glg_hs", "outputs_plots/hs-cosine-zero_one_tiny_glg/stdout")
  plt.clf()
  prob_plot("glg_uni", "outputs_plots/uniform-cosine-zero_one_tiny_glg/stdout")
  plt.clf()
  prob_plot("slk_hs", "outputs_plots/hs-cosine-zero_one_tiny_slk/stdout")
  plt.clf()
  prob_plot("slk_uni", "outputs_plots/uniform-cosine-zero_one_tiny_slk/stdout")

