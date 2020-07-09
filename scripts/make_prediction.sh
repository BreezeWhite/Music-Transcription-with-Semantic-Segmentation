#!/usr/bin/env python3

import os
import sys
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from project.Evaluate.Evaluation import EvalEngine

feature_path = "/data/Maps/test_feature/harmonic"
#feature_path = "/host/home/AMT_Project/Maestro-val_feature/harmonic"
model_path = "../model-paper/feature-compare_harmonic"
pred_save_path = "../prediction-paper/feature-compare_harmonic"

generator = EvalEngine.predict_dataset(feature_path, model_path, pred_save_path)

ii = 0
for pred, ll, key in generator:
    ii += 1
    print(ii, pred.shape, key)
