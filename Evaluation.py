
import os
import h5py
import argparse
import numpy as np

from project.Evaluate.predict import predict
from project.Evaluate.evaluation import evaluation
from project.Dataflow import DataFlows
from project.Dataflow.evalflow import EvalFlow
from project.utils import load_model, model_info
from TrainModel import dataset_paths, dataflow_cls


def save_pred(preds, labels, out_path):
    ff = h5py.File(os.path.join(out_path, "pred.hdf"), "w")
    ll = h5py.File(os.path.join(out_path, "label.hdf"), "w")

    for i in range(len(preds)):
        ff.create_dataset(str(i), data=preds[i], compression="gzip", compression_opts=5)
        ll.create_dataset(str(i), data=labels[i], compression="gzip", compression_opts=5)

    ff.close()
    ll.close()

def main(args):
    
    model = load_model(args.model_path)
    feature_type, channels, out_classes, timesteps = model_info(args.model_path)
    
    d_path = dataset_paths[args.dataset]
    df_cls = dataflow_cls[args.dataset]
    df = df_cls(d_path, "test", timesteps=timesteps, channels=channels, b_sz=16)
    eval_flow = EvalFlow(df)

    
    preds = []
    lls = []
    for _ in range(len(eval_flow)):
        # This loop go through pieces
        features = []
        labels = []
        for x, y in eval_flow:
            # Collect batches from single piece
            features.append(x)
            labels.append(y)

        pred, ll = predict(features, labels, model)

        preds.append(pred)
        lls.append(ll)
    
    precision, recall, f1, l_prec, l_rec, l_f = Evaluation(preds, lls)
    print("Prec: {:.4f}  Recall: {:.4f}  F1: {:.4f}".format(precision, recall, f1))

    if args.save_pred is not None:
        if not os.path.exists(args.save_pred):
            os.makedirs(args.save_pred)
        save_pred(preds, lls, args.save_pred)
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation program for this project")
    
    parser.add_argument("dataset", help="One of the Maestro, MusicNet, or Maps", type=str)
    parser.add_argument("model_path", help="Path to the directory of trained model", type=str)
    parser.add_argument("-s", "--save-pred", help="Wether to store the prediction to given paths",
                        type=str, default=None)
    
    args = parser.parse_args()
    main(args)
    
