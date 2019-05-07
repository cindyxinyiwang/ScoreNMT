
TEMP_DIR=scripts/template/
# change random seed and directory name as desired
CFG_DIR=cfg_s2/
SEED=2
DATA=tiny
DATA_DIR='\/home\/hyhieu\/xinyiw\/ScoreNMT\/'
THRESH=15

mkdir -p scripts/"$CFG_DIR"
# low-resource language codes
ILS=(
  bel)
#  glg)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  echo $IL
  echo $DATA
  #for f in  $TEMP_DIR/uniform-cosine-zero_one $TEMP_DIR/uniform-cosine-one $TEMP_DIR/uniform $TEMP_DIR/cur-cosine-zero_one $TEMP_DIR/cur $TEMP_DIR/hs $TEMP_DIR/hs-cosine-lan_dist $TEMP_DIR/hs-cosine-one $TEMP_DIR/hs-cosine-zero_one; do
  for f in $TEMP_DIR/hs $TEMP_DIR/hs-cosine-zero_one ; do
    sed "s/DATA_DIR/$DATA_DIR/g; s/THRESH/$THRESH/g; s/DATA/$DATA/g; s/SEED/$SEED/g; s/IL/$IL/g;" < $f > ${f/template/"$CFG_DIR"/}_"$DATA"_$IL.sh 
    chmod u+x ${f/template/"$CFG_DIR"/}_"$DATA"_$IL.sh 
  done
done
