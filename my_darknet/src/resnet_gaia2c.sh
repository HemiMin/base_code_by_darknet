#!/bin/bash

layers=("conv_1" \
"conv_1_0_1" \
"conv_1_0_2" \
"conv_1_0_3" \
"conv_1_1_1" \
"conv_1_1_2" \
"conv_1_1_3" \
"conv_1_2_1" \
"conv_1_2_2" \
"conv_1_2_3" \
"conv_2_0_1" \
"conv_2_0_2" \
"conv_2_0_3" \
"conv_2_1_1" \
"conv_2_1_2" \
"conv_2_1_3" \
"conv_2_2_1" \
"conv_2_2_2" \
"conv_2_2_3" \
"conv_2_3_1" \
"conv_2_3_2" \
"conv_2_3_3" \
"conv_3_0_1" \
"conv_3_0_2" \
"conv_3_0_3" \
"conv_3_1_1" \
"conv_3_1_2" \
"conv_3_1_3" \
"conv_3_2_1" \
"conv_3_2_2" \
"conv_3_2_3" \
"conv_3_3_1" \
"conv_3_3_2" \
"conv_3_3_3" \
"conv_3_4_1" \
"conv_3_4_2" \
"conv_3_4_3" \
"conv_3_5_1" \
"conv_3_5_2" \
"conv_3_5_3" \
"conv_4_0_1" \
"conv_4_0_2" \
"conv_4_0_3" \
"conv_4_1_1" \
"conv_4_1_2" \
"conv_4_1_3" \
"conv_4_2_1" \
"conv_4_2_2" \
"conv_4_2_3" \
)

hw=$1
version=(is ws os opt)

for v in ${version[@]}
do
  GAIA_DIR=~/workspace/gaia.code/resnet50.$hw.$v.output
  TARGET_DIR=jetson/resnet50.$hw.$v.srcs
  echo $GAIA_DIR
  echo $TARGET_DIR

  mkdir $TARGET_DIR
  for i in "${layers[@]}";
  do
    python gaia_parser.py -o $TARGET_DIR/$i.c -c $GAIA_DIR/$i.gaia;
  done
done
