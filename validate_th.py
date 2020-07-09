
import numpy as np
from project.Evaluate.Evaluation import EvalEngine 

if __name__ == "__main__":
    feature_type = "spec+ceps+gcos"
    pred_path = f"prediction-paper/val_feature-compare_{feature_type}/feature-compare_{feature_type}_predictions.hdf"
    label_path = f"prediction-paper/val_feature-compare_{feature_type}/feature-compare_{feature_type}_labels.pickle"

    mode = "note"
    step = np.arange(0.25, 0.5, 0.05)
    best_fs = 0
    best_th = step[0]
    for idx, i in enumerate(step, 1):
        print("{}/{} Onset threshold: {}".format(idx, len(step), i))
        eval_results = EvalEngine.evaluate_dataset(
            mode,
            pred_path=pred_path, 
            label_path=label_path,
            onset_th=i
        )
        prec, rec, fs, _, _ = eval_results.get_avg()

        if fs > best_fs:
            best_fs = fs
            best_th = i

    print("Best threshold: {}, F-score: {:.4f}".format(best_th, best_fs))
