#!/bin/bash

out_model=Maestro-Maps-Attn-Note-Smooth
epoch=10
early_stop=6

cd ..
python3 TrainModel.py Maps $out_model \
    --epoch $epoch           \
    --steps 3000             \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8       \
    -i ./model/Maestro-Attn-Note-Smooth
