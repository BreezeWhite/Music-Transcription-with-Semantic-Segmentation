
import numpy as np

from scipy.special import expit
from project.Evaluation.eval_utils import cut_frame, cut_batch_pred


def predict(features, model, labels=None):
    """
        features dim: batches x batch_size x timesteps x feature_size x channels
    """
    pred = []
    label = []

    cut_frm = lambda x: cut_frame(x, ori_feature_size=352, feature_num=features[0][0].shape[1])

    t_len = len(features[0][0])
    first_split_start = round(t_len*0.75)
    second_split_start = t_len+round(t_len*0.25)
    
    total_batches = len(features)
    features.insert(0, [np.zeros_like(features[0][0])])
    features.append([np.zeros_like(features[0][0])])
    for i in range(1, total_batches+1):
        print("batch: {}/{}".format(i, total_batches), end="\r")

        first_half_batch = []
        second_half_batch = []
        b_size = len(features[i])
        features[i] = np.insert(features[i], 0, features[i-1][-1], axis=0)
        features[i] = np.insert(features[i], len(features[i]), features[i+1][0], axis=0)
        for ii in range(1, b_size+1):
            ctx = np.concatenate(features[i][ii-1:ii+2], axis=0) 

            first_half = ctx[first_split_start:first_split_start+t_len]
            first_half_batch.append(first_half)

            second_half = ctx[second_split_start:second_split_start+t_len]
            second_half_batch.append(second_half)

        p_one = model.predict(np.array(first_half_batch), batch_size=b_size)
        p_two = model.predict(np.array(second_half_batch), batch_size=b_size)
        p_one = cut_batch_pred(p_one)
        p_two = cut_batch_pred(p_two)
        
        for ii in range(b_size):
            frm = np.concatenate([p_one[ii], p_two[ii]])
            pred.append(cut_frm(frm))
            if labels is not None:
                label.append(cut_frm(labels[i-1][ii]))

    pred = expit(np.concatenate(pred)) # sigmoid function, mapping the ReLU output value to [0, 1]
    if labels is not None:
        label = np.concatenate(label)
        return pred, label
        
    return pred

