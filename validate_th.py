
import numpy as np
from project.Evaluate.Evaluation import EvalEngine 

if __name__ == "__main__":
    pred_path = "/media/data/maps/val_feature/maestro_prediction/Maestro-Attn-W4.2_predictions.hdf"
    label_path = "/media/data/maps/val_feature/maestro_prediction/Maestro-Attn-W4.2_labels.pickle"

    pred_path = "/media/data/maps/val_feature/prediction/Maps-Attn-W4.2.1_predictions.hdf"
    label_path = "/media/data/maps/val_feature/prediction/Maps-Attn-W4.2.1_labels.pickle"

    pred_path = "/host/home/AMT_Project/prediction/val_musicnet_smooth_v1.0.2/MusicNet-Attn-Note-Smooth-V1.0.2_predictions.hdf"
    label_path = "/host/home/AMT_Project/prediction/val_musicnet_smooth_v1.0.2/MusicNet-Attn-Note-Smooth-V1.0.2_labels.pickle"
    
    step = np.arange(7.5, 10, 0.5)
    best_fs = 0
    best_th = step[0]
    for idx, i in enumerate(step, 1):
        print("{}/{} Onset threshold: {}".format(idx, len(step), i))
        prec, rec, fs = EvalEngine.evaluate_dataset(
            "note",
            pred_path=pred_path, 
            label_path=label_path,
            onset_th=i
        )

        if fs > best_fs:
            best_fs = fs
            best_th = i

    print("Best threshold: {}, F-score: {:.4f}".format(best_th, best_fs))
