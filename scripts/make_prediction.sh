#!/usr/bin/env python3

import os
import sys
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from project.Evaluate.Evaluation import EvalEngine

feature_path = "/data/MusicNet/test_feature"
model_path = "../model/Dilated-Conv-MusicNet-Note-Smooth"
pred_save_path = "../prediction/dilated_conv_musicnet_note_smooth"

generator = EvalEngine.predict_dataset(feature_path, model_path, pred_save_path)

ii = 0
for pred, ll, key in generator:
    ii += 1
    print(ii, pred.shape, key)
