#!/bin/bash
file_list=(ops.h scat_gather_gpu.cu scat_gather.h conv_gpu.cu timer_gpu.cu malloc_gpu.cu "../gaia_parser.py")

conversion=$1

if [ $conversion = 2 ];then
  bak=bak2
elif [ $conversion = 1 ];then
  bak=bak
fi

for f in ${file_list[@]}
do
  cp $f.$bak $f
done

cp alexnet_c.$bak/* .

cd ..
./gaia2c.sh

