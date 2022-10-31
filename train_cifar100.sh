#!/bin/bash
# training

out=./training/standard/
dataset="cifar100"
arch="resnet18"
transform_type="semantic spatial color blur"

for transform in $transform_type
do
echo $transform
out_dir=$out$dataset/$arch/$transform/aug
echo $out_dir
echo $dataset
echo $arch

CUDA_VISIBLE_DEVICES=3 python -m robustness.main \
       --dataset $dataset \
       --epochs 105\
       --adv-train 0 \
       --adv-eval 1 \
       --data-aug 1 \
       --tries 10 \
       --use-best 1 \
       --rot 30\
       --trans 0.3\
       --scale 0.3\
       --hue 1.5707963268\
       --satu 0.3\
       --bright 0.3\
       --cont 0.3\
       --gau-size 11\
       --gau-sigma 9\
       --transform-type $transform \
       --attack-type "random" \
       --out-dir $out_dir \
       --arch $arch \
       --data "./path/to/cifar" \
       --batch-size 128
done






