#!/bin/bash

out_model=full_model_all_feature_musicnet
epoch=15
early_stop=6

cd ..
python3 TrainModel.py MusicNet $out_model \
    --channels 1 2 3         \
    --epoch $epoch           \
    --steps 3000             \
    --timesteps 128          \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8       
    #--use-harmonic
    #-i ./model/Maestro-Smooth-Ultimate-Attn-V1

