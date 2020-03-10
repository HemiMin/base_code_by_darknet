#!/bin/bash

env=(env1 env2 env5 env6)
net=(alexnet vgg16 resnet squeezenet yolov2)
#net=(yolov2)
for e in ${env[@]}
do
for n in ${net[@]}
do
  ./${n}_gaia2c.sh ${e}
done
done

