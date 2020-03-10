#!/bin/bash

#net=(squeezenet resnet50 vgg16)
net=(alexnet)
version=(is ws os opt)
#version=(is)
env=(env1 env2 env5 env6)
#env=(env2 env5 env6)
#target=(CUDA CUBLAS)
target=(CUBLAS)


for t in ${target[@]}
do
  for e in ${env[@]}
  do
    for n in ${net[@]}
    do
      nvprof_dirs=nvprof_files/$n
      mkdir $nvprof_dirs 
      for v in ${version[@]}
      do
        resname=
        if [ ${t} = "CUBLAS" ];then
          resname=$nvprof_dirs/$n.$e.$v.nvprof
        fi
        if [ ${t} = "CUDA" ];then
          resname=$nvprof_dirs/$n.$e.$v.cuda.nvprof
        fi
        nvprof_cmd="nvprof -f --analysis-metrics -o $resname "

        if [ ${t} = "CUBLAS" ];then
          $nvprof_cmd ./$n.$e.$v ../../data/dog.jpg ../../data/$n.weights 5
        fi
        if [ ${t} = "CUDA" ];then
          $nvprof_cmd ./$n.$e.$v.cuda ../../data/dog.jpg ../../data/$n.weights 5
        fi

      done

    done
  done
done
