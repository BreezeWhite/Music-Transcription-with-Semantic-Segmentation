#!/bin/bash

out_model=Maestro-Smooth-Ultimate
epoch=10
early_stop=6

cd ..
python3 TrainModel.py Maestro $out_model \
    --epoch $epoch           \
    --steps 3000             \
    --timesteps 512          \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8       
    #-i ./model/Maestro-Smooth-Ultimate-Attn
