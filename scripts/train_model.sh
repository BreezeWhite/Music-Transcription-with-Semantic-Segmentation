#!/bin/bash

out_model=Maestro-Attn-W4.2.1
epoch=12
early_stop=6

cd ..
python3 TrainModel.py Maestro $out_model \
    --epoch $epoch           \
    --steps 5000             \
    --early-stop $early_stop \
    --val-batch-size 8       
    #-i ./model/Maestro-Attn-V4.1
