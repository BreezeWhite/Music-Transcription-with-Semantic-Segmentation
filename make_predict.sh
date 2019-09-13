#!/bin/bash

model=./model/Maestro-Onset-Offset
feature=/media/whitebreeze/本機磁碟/maps/test_feature
label=/media/whitebreeze/本機磁碟/maps/test_label
save_path=./predictions/Maestro-Onset-Offset

python3 Predict.py --model-path $model \
                   --test-path $feature \
                   --label-path $label \
                   --pred-save-path $save_path

#python3 Evaluation Maestro $model -s $save_path
