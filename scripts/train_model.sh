#!/bin/bash

out_model=MusicNet-Attn-W4.2
epoch=12
early_stop=6

cd ..
python3 TrainModel.py MusicNet $out_model \
    --epoch $epoch           \
    --steps 5000             \
    --early-stop $early_stop \
    --val-batch-size 8       
    #-i ./model/Maestro-Attn-V4.1
