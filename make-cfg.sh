
TEMP_DIR=scripts/template/
# change random seed and directory name as desired
CFG_DIR=cfg_s1/
SEED=1
DATA=mod

mkdir -p scripts/"$CFG_DIR"
# low-resource language codes
ILS=(
  slv)
#  glg)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  echo $IL
  echo $DATA
  #for f in $TEMP_DIR/hs $TEMP_DIR/hs-cosine-lan_dist $TEMP_DIR/hs-cosine-lan_dist-v2 $TEMP_DIR/hs-cosine-one ; do
  for f in $TEMP_DIR/hs $TEMP_DIR/uniform-cosine-zero_one; do
    sed "s/DATA/$DATA/g; s/SEED/$SEED/g; s/IL/$IL/g;" < $f > ${f/template/"$CFG_DIR"/}_"$DATA"_$IL.sh 
    chmod u+x ${f/template/"$CFG_DIR"/}_"$DATA"_$IL.sh 
  done
done
