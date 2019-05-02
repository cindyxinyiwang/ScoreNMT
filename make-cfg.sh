
TEMP_DIR=scripts/template/
# change random seed and directory name as desired
CFG_DIR=cfg_s1/
SEED=1

mkdir -p scripts/"$CFG_DIR"
# low-resource language codes
ILS=(
  aze
  glg)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  echo $IL
  for f in $TEMP_DIR/hs $TEMP_DIR/hs-cosine-lan_dist $TEMP_DIR/uniform-cosine-zero_one; do
    sed "s/SEED/$SEED/g; s/IL/$IL/g;" < $f > ${f/template/"$CFG_DIR"/}_$IL.sh 
    chmod u+x ${f/template/"$CFG_DIR"/}_$IL.sh 
  done
done
