

# combine data into src:[L1, L2...Ln], trg:[eng]

data_dir = "/home/xinyiw/multv-nmt/"
#data_dir = "/projects/tir1/users/xinyiw1/multv-nmt/"
out_dir = "data/big/"
lang_file = "langs_big.txt"
out_src = out_dir + "ted-train.mtok.spm8000.big"
out_trg = out_dir + "ted-train.mtok.spm8000.eng"

lans = []
with open(lang_file, 'r') as myfile:
  for line in myfile:
    lans.append(line.strip())
lan_size = len(lans)

print(lans)
data = {}
for i, l in enumerate(lans):
  src = data_dir + "data/{}_eng/ted-train.mtok.spm8000.{}".format(l, l)
  trg = data_dir + "data_moses/{}_eng/ted-train.mtok.spm8000.eng".format(l)
  src_file = open(src, 'r')
  trg_file = open(trg, 'r')
  for s, t in zip(src_file, trg_file):
    if not t in data:
      data[t] = ["\n" for _ in range(lan_size)]
    data[t][i] = s

out_src_file = open(out_src, 'w')
out_trg_file = open(out_trg, 'w')
for t, src_list in data.items():
  out_trg_file.write(t)
  for s in src_list:
    out_src_file.write(s)
  out_src_file.write("EOF\n")
out_src_file.close()
out_trg_file.close()
