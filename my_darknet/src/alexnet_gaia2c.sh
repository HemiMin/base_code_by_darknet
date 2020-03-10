#!/bin/bash

layers=("conv_1" \
"conv_2" \
"conv_3" \
"conv_4" \
"conv_5"
)

env=$1
version=(is ws os opt)

for v in ${version[@]}
do
  GAIA_DIR=~/workspace/gaia.code/alexnet.$env.$v.output
  TARGET_DIR=jetson/alexnet.$env.$v.srcs
  echo $GAIA_DIR
  echo $TARGET_DIR

  mkdir $TARGET_DIR
  for i in "${layers[@]}"
  do
      python gaia_parser.py -o $TARGET_DIR/$i.c -c $GAIA_DIR/$i.gaia;
  done
done
