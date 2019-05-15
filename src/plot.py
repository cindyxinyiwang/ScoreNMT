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
      p = [float(i) for i in p.split(", ")]
      cur_probs.append(p)
  return lan_probs_list

def prob_plot(name, stdout):
  linewidth = 5.0
  markersize = 15

  probs_list = get_probs(stdout)
  final_probs_tur = []
  final_probs_rus = []
  final_probs_por = []
  final_probs_ces = []
  for prob in probs_list:
    ave_prob = np.sum(prob, axis=0) / np.sum(prob > 0, axis=0)
    final_probs_tur.append(ave_prob[4])
    final_probs_rus.append(ave_prob[5])
    final_probs_por.append(ave_prob[6])
    final_probs_ces.append(ave_prob[7])
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
  prob_plot("aze", "outputs_s2/uniform-cosine-zero_one-v2_tiny_aze/stdout")

