#!/bin/bash
# training

out=./training/aug/
dataset="cifar cifar100 restricted_imagenet"
arch="resnet18"
transform_type="semantic spatial color blur"
aug="RandAugment TrivialAugment AugMix"
for transform in $transform_type
do
for data in $dataset
do
for aug_method in $aug
do

echo $transform
echo $data
echo $aug_method
out_dir=$out$data/$arch/$aug_method/$transform
echo $out_dir
echo $arch

CUDA_VISIBLE_DEVICES=0,1 python -m robustness.main \
       --dataset $data \
       --epochs 105\
       --adv-train 0 \
       --adv-eval 1 \
       --data-aug 1 \
       --aug-method $aug_method \
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
done
done




