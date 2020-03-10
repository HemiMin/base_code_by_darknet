#!/bin/bash

layers=("conv_1_1" \
"conv_1_2" \
"conv_2_1" \
"conv_2_2" \
"conv_3_1" \
"conv_3_2" \
"conv_3_3" \
"conv_4_1" \
"conv_4_2" \
"conv_4_3" \
"conv_5_1" \
"conv_5_2" \
"conv_5_3"
)

hw=$1
version=(is ws os opt)

for v in ${version[@]}
do
  GAIA_DIR=~/workspace/gaia.code/vgg16.$hw.$v.output
  TARGET_DIR=jetson/vgg16.$hw.$v.srcs
  echo $GAIA_DIR
  echo $TARGET_DIR

  mkdir $TARGET_DIR
  for i in "${layers[@]}";
  do
    python gaia_parser.py -o $TARGET_DIR/$i.c -c $GAIA_DIR/$i.gaia;
  done
done
