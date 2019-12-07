#!/bin/bash

out_model=ICASSP-2019-Maestro-Frame
epoch=10
early_stop=6

cd ..
python3 TrainModel.py Maestro $out_model \
    --epoch $epoch           \
    --steps 3000             \
    --early-stop $early_stop \
    --train-batch-size 16    \
    --val-batch-size 16      \
    -i ./model/ICASSP-2019-Maestro-Frame
