#!/bin/bash
# training

out=./training/adv/
dataset="restricted_imagenet"
arch="resnet18"
transform_type="color blur"

for transform in $transform_type
do
echo $transform
out_dir=$out$dataset/$arch/$transform/no_aug
echo $out_dir
echo $dataset
echo $arch

CUDA_VISIBLE_DEVICES=0,3 python -m robustness.main \
       --dataset $dataset \
       --epochs 110\
       --adv-train 1 \
       --adv-eval 1 \
       --data-aug 0 \
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
       --data "/datasets/ImageNet2012" \
       --batch-size 64 \
       --subset 76800
done
# python -m robustness.main \
#        --dataset "cifar" \
#        --epochs 110\
#        --adv-train 1 \
#        --adv-eval 1 \
#        --data-aug 1 \
#        --tries 1 \
#        --use-best 0 \
#        --rot 30\
#        --trans 0.3\
#        --scale 0.3\
#        --hue 1.5707963268\
#        --satu 0.3\
#        --bright 0.3\
#        --cont 0.3\
#        --gau-size 11\
#        --gau-sigma 9\
#        --transform-type 'semantic' \
#        --attack-type "random" \
#        --out-dir ./training/adv/cifar/resnet18/semantic/ \
#        --arch 'resnet18' \
#        --data "./path/to/cifar" \
#        --batch-size 128





