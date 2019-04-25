
import numpy as np

def cut_frame(frm, ori_feature_size=352, feature_num=384):
    cb = (feature_num-ori_feature_size) // 2
    c_range = range(cb, cb+ori_feature_size)
    
    return frm[:, c_range]


def predict(features, labels, model):
    pred = []
    label = []

    cut_frm = lambda x: cut_frame(x, ori_feature_size=352, feature_num=df.feature_num)
    
    total_batches = len(features)
    for i in range(total_batches):
        ff = features[i]
        ll = labels[i]

        p = model.predict(ff, batch_size=ff.shape[0])
        for idx, pp in enumerate(p):
            pred.append(cut_frm(pp))
            label.append(cut_frm(ll[idx]))

    pred = np.concatenate(pred)
    label = np.concatenate(label)

    return pred, label

