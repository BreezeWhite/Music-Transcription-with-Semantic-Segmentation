#!/bin/bash

out_model=feature-compare_harmonic
epoch=15
early_stop=6
#label_type=multi_instrument_note
label_type=frame_onset

cd ..
python3 TrainModel.py Maestro $out_model \
    --channels 1 2 3         \
    --label-type $label_type \
    --epoch $epoch           \
    --steps 3000             \
    --timesteps 128          \
    --early-stop $early_stop \
    --train-batch-size 8     \
    --val-batch-size 8       \
    --use-harmonic
    #-i ./model/Maestro-Smooth-Ultimate-Attn-V1

