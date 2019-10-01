#!/usr/bin/env python3

import os
import sys
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from project.Evaluate.Evaluation import EvalEngine

feature_path = "/data/Maps/test_feature2"
model_path = "../model/Maestro-Attn-W4.2"
pred_save_path = feature_path + "/maestro_prediction"
#pred_save_path = "./prediction/maps_old_result"

generator = EvalEngine.predict_dataset(feature_path, model_path, pred_save_path)

ii = 0
for pred, ll, key in generator:
    ii += 1
    print(ii, pred.shape, key)
