#!/bin/bash

net=(alexnet squeezenet resnet50 yolov2 )
#net=(alexnet squeezenet resnet50 vgg16)
#net=(vgg16)
version=(os is ws opt)
#version=(is)
repeat=1
env=(env1 env2 env5 env6)
#env=(env2 )
#target=(CUDA CUBLAS)
target=(CUBLAS)

timeest=1

for t in ${target[@]}
do
  for e in ${env[@]}
  do
    for n in ${net[@]}
    do

      for v in ${version[@]}
      do
        resname=
        if [ ${t} = "CUBLAS" ];then
          if [ $timeest = 1 ];then
            resname=./results/$n.$e.$v.timeest.res
          else
            resname=./results/$n.$e.$v.res
          fi
        fi
        if [ ${t} = "CUDA" ];then
          if [ $timeest = 1 ];then
            resname=./results/$n.$e.$v.timeest.res.cuda
          else
            resname=./results/$n.$e.$v.res.cuda
          fi
        fi
        rm $resname
        total_time=0.0

        avg_time=0.0
        echo $n.$e.$v >> $resname

        if [ $timeest = 1 ];then
          make_option="$t=1 net=$n hw=$e version=$v TIME_ESTIMATE=1"
          bin_file=./$n.$e.$v.timeest
        else
          make_option="$t=1 net=$n hw=$e version=$v TIME_ESTIMATE=0"
          bin_file=./$n.$e.$v
        fi
          cmd="make -j6 $make_option"
          echo $cmd
          eval $cmd

        for i in $(seq 1 $repeat)
        do
          echo ${i}_repeat >> $resname
            if [ ${t} = "CUBLAS" ];then
              if [ $i = $repeat ];then
                if [ $n = yolov2 ];then
                  cmd="$bin_file ../../data/dog.jpg ../../data/$n.weights | tee -a $resname"
                else
                  cmd="$bin_file ../../data/dog.jpg ../../data/$n.weights 5 | tee -a $resname"
                fi
              else
                if [ $n = yolov2 ];then
                  cmd="$bin_file ../../data/dog.jpg ../../data/$n.weights | tee -a $resname"
                else
                  cmd="$bin_file ../../data/dog.jpg ../../data/$n.weights 5 | tee -a $resname"
                fi
#cmd="./$n.$e.$v ../../data/dog.jpg ../../data/$n.weights 5"
              fi
#result=$(./$n.$e.$v ../../data/dog.jpg ../../data/$n.weights 5)
            elif [ ${t} = "CUDA" ];then
              if [ $i = $repeat ];then
                cmd="$bin_file.cuda ../../data/dog.jpg ../../data/$n.weights 5 | tee -a $resname"
              else
                cmd="$bin_file.cuda ../../data/dog.jpg ../../data/$n.weights 5 | tee -a $resname"
#cmd="./$n.$e.$v.cuda ../../data/dog.jpg ../../data/$n.weights 5"
              fi
#result=$(./$n.$e.$v.cuda ../../data/dog.jpg ../../data/$n.weights 5)
            fi
            echo $cmd
            result=$(eval $cmd)
#time=$(echo $result | cut -f2 -d "m" | cut -f2 -d ":")
            time=$(echo $result | cut -f 2 -d "E" | cut -f 2 -d "m" | cut -f 2 -d ":")
            echo $time
            total_time=$(echo $total_time+$time|bc)
            echo " " >> $resname
        done # repeat done

            avg_time=$(echo $total_time/$repeat.0|bc -l)
            echo avg_time: $avg_time >> $resname
            echo " " >> $resname

      done # version done

    cmd="make $make_option clean"
    echo $cmd
    eval $cmd

    done
  done
done
