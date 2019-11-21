#!/bin/bash

out_model=MusicNet-Attn-Note-Smooth-V1.0.2
epoch=10
early_stop=6

cd ..
python3 TrainModel.py MusicNet $out_model \
    --epoch $epoch           \
    --steps 5000             \
    --early-stop $early_stop \
    --val-batch-size 8       \
    -i ./model/MusicNet-Attn-Note-Smooth-V1.0.1
