#!/bin/bash

layers=("conv_0" \
"sqz_1" \
"exp_1_1" \
"exp_1_3" \
"sqz_2" \
"exp_2_1" \
"exp_2_3" \
"sqz_3" \
"exp_3_1" \
"exp_3_3" \
"sqz_4" \
"exp_4_1" \
"exp_4_3" \
"sqz_5" \
"exp_5_1" \
"exp_5_3" \
"sqz_6" \
"exp_6_1" \
"exp_6_3" \
"sqz_7" \
"exp_7_1" \
"exp_7_3" \
"sqz_8" \
"exp_8_1" \
"exp_8_3" \
"conv_c"
)

hw=$1
version=(is ws os opt)

for v in ${version[@]}
do
  GAIA_DIR=~/workspace/gaia.code/squeezenet.$hw.$v.output
  TARGET_DIR=jetson/squeezenet.$hw.$v.srcs
  echo $GAIA_DIR
  echo $TARGET_DIR

  mkdir $TARGET_DIR
  for i in "${layers[@]}";
  do
    python gaia_parser.py -o $TARGET_DIR/$i.c -c $GAIA_DIR/$i.gaia;
  done
done
