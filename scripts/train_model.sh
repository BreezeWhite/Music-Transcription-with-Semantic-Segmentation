#!/bin/bash

out_model=MusicNet-Attn-Note-Smooth-V1.0.2
out_model=ICASSP-2019-MusicNet-Note
out_model=Dilated-Conv-MusicNet-Note-Smooth
epoch=10
early_stop=6

cd ..
python3 TrainModel.py MusicNet $out_model \
    --epoch $epoch           \
    --steps 2500             \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8      
    #-i ./model/Dilated-Conv-MusicNet-Note-Smooth
