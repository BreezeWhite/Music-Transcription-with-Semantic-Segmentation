#!/bin/bash

out_model=Dilated-Conv-Maestro-Note-Smooth
epoch=15
early_stop=6

cd ..
python3 TrainModel.py Maestro $out_model \
    --epoch $epoch           \
    --steps 3000             \
    --timesteps 128          \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8       
    #--use-harmonic
    #-i ./model/Maestro-Smooth-Ultimate-Attn-V1
