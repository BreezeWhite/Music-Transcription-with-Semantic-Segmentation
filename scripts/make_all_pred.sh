#!/usr/bin/env python3

import os
import sys
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from project.Evaluate.Evaluation import EvalEngine

feature_path = "/data/Maps/test_feature"
pred_save_path = "../prediction-paper/feature_compare_spec-gcos"
model_path = [
    "../model-paper/feature-compare_spec-only",
    "../model-paper/feature-compare_spec+ceps",
    "../model-paper/feature-compare_ceps+gcos",
    "../model-paper/feature-compare_spec+ceps+gcos",
    "../model-paper/feature-compare_ceps-only",
    "../model-paper/feature-compare_gcos-only",
    "../model-paper/feature-compare_spec+gcos"
]

for model in model_path:
    pred_save_path = "../prediction-paper/"+model.split("/")[-1]
    generator = EvalEngine.predict_dataset(feature_path, model, pred_save_path)

    ii = 0
    for pred, ll, key in generator:
        ii += 1
        print(ii, pred.shape, key)
