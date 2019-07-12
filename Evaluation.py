
import os
import h5py
import argparse
import numpy as np

from project.Evaluate.predict import predict
from project.Evaluate.evaluation import evaluation
from project.Evaluate.eval_utils import save_pred, roll_down_sample
from project.Dataflow.evalflow import EvalFlow
from project.utils import load_model, model_info
from TrainModel import dataset_paths, dataflow_cls


def eval_stats(l_prec, l_rec, l_f):
    precision = sum(l_prec) / len(l_prec)
    recall = sum(l_rec) / len(l_rec)
    f1 = sum(l_f) / len(l_f)

    print("Middle evaluation")
    print("Prec: {:.4f}  Recall: {:.4f}  F1: {:.4f}".format(precision, recall, f1))
    
    return l_prec, l_rec, l_f

def main(args):
    
    model = load_model(args.model_path)
    feature_type, channels, out_classes, timesteps = model_info(args.model_path)
    
    d_path = dataset_paths[args.dataset]
    df_cls = dataflow_cls[args.dataset]
    df = df_cls(d_path, "test", timesteps=timesteps, channels=channels, b_sz=16)
    eval_flow = EvalFlow(df)

    
    wr_f = None
    wr_l = None
    if args.save_pred is not None:
        if not os.path.exists(args.save_pred):
            os.makedirs(args.save_pred)
        out_f = h5py.File(os.path.join(args.save_pred, "pred.hdf"), "w")
        out_l = h5py.File(os.path.join(args.save_pred, "label.hdf"), "w")

        wr_f = lambda i, d: out_f.create_dataset(str(i), data=d, compression="gzip", compression_opts=5)
        wr_l = lambda i, l: out_l.create_dataset(str(i), data=l, compression="gzip", compression_opts=5)

    
    preds = []
    lls = []
    results = {"l_prec": [], "l_rec": [], "l_f": []}
    for i in range(10):#len(eval_flow)):
        # This loop go through pieces
        print("{}/{}".format(i+1, len(eval_flow)))

        features = []
        labels = []
        for x, y in eval_flow:
            # Collect batches from a single piece
            features.append(x)
            labels.append(y)
            #print(y.shape)

        pred, ll = predict(features, labels, model)
        
        """
        p = np.where(pred[:,:,1]>pred[:,:,0], 1, 0)
        l = ll[:,:,1]
        prec, rec, f, l_prec, l_rec, l_f = evaluation([p], [ll])
        results["l_prec"] += l_prec
        results["l_rec"]  += l_rec
        results["l_f"]    += l_f

        if len(preds)%2 == 0:
            eval_stats(l_prec, l_rec, l_f)
        """

        if args.save_pred is not None:
            wr_f(i, pred)
            wr_l(i, ll)

#for i in range(len(preds)):
    #    p = preds[i]
    #    a = np.where(p[:,:,1]>p[:,:,0], 1, 0)
    #    preds[i] = roll_down_sample(a)
    #    lls[i] = roll_down_sample(lls[i])

    #eval_stats(l_prec, l_rec, l_f)

    if args.save_pred is not None:
        out_f.close()
        out_l.close()
    
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation program for this project")
    
    parser.add_argument("dataset", help="One of the Maestro, MusicNet, or Maps", type=str)
    parser.add_argument("model_path", help="Path to the directory of trained model", type=str)
    parser.add_argument("-s", "--save-pred", help="Wether to store the prediction to given paths",
                        type=str, default=None)
    
    args = parser.parse_args()
    main(args)
    
