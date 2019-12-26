import numpy as np
from scipy.special import expit

def predict(feature, model, timesteps, out_class, feature_num=384, batch_size=4, overlap_ratio=0.5):
    step_size = round((1-overlap_ratio)*timesteps)
    if step_size == 0:
        raise ValueError("Given overlap_ratio too high. Please decrease the value (available: [0, 1))")
    
    steps = np.ceil(len(feature)/step_size).astype('int')
    steps += np.ceil(1/overlap_ratio).astype('int') - 1
    
    # Padding height
    pb = (feature_num-feature.shape[1]) // 2
    pt = feature_num-feature.shape[1]-pb
    length = len(feature)
    channel = feature.shape[2]
    pbb = np.zeros((length, pb, channel))
    ptt = np.zeros((length, pt, channel))
    feature = np.hstack([pbb, feature, ptt])
    
    # Padding length
    pad_begin_len = round(overlap_ratio*timesteps)
    pad_begin = np.zeros((pad_begin_len, feature_num, feature.shape[2]))
    pad_to_multiple = step_size - (length%step_size)
    pad_end_len = pad_to_multiple + pad_begin_len + (batch_size-steps%batch_size)*step_size
    pad_end = np.zeros((pad_end_len, feature_num, feature.shape[2]))
    feature = np.concatenate([pad_begin, feature, pad_end])
   

    # Some other parameters
    batch_num = np.ceil(steps/batch_size).astype('int')
    batch = np.zeros((batch_size, timesteps,) + feature.shape[1:])
    hop_size = batch_size*step_size
    pred_buffer = np.zeros(feature.shape[:-1]+(out_class,))
    
    # Start prediction
    for i in range(batch_num):
        print("{}/{}".format(i+1, batch_num), end="\r")
        
        # Collect batch
        start_frm = i*hop_size
        for b in range(batch_size):
            frm_range = range(start_frm+b*step_size, start_frm+b*step_size+timesteps)
            batch[b] = feature[frm_range]

        pred = model.predict(batch, batch_size=batch_size)
        
        # Add prediction to buffer
        for b in range(batch_size):
            frm_range = range(start_frm+b*step_size, start_frm+b*step_size+timesteps)
            pred_buffer[frm_range] += pred[b]
        
    return expit(pred_buffer[pad_begin_len:pad_begin_len+length, pb:feature_num-pt] * (1-overlap_ratio))


def predict_v1(feature, model, timesteps, feature_num=384, batch_size=4, labels=None):
    """
        features dim: batches x batch_size x timesteps x feature_size x channels
    """
    features = create_batches(feature, b_size=batch_size, timesteps=timesteps, feature_num=feature_num) 


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


def cut_frame(frm, ori_feature_size=352, feature_num=384):
    feat_num = frm.shape[1]
    assert(feat_num==feature_num)

    cb = (feat_num-ori_feature_size) // 2
    c_range = range(cb, cb+ori_feature_size)
    
    return frm[:, c_range]

def cut_batch_pred(b_pred):
    t_len = len(b_pred[0])
    cut_rr = range(round(t_len*0.25), round(t_len*0.75))
    cut_pp = []
    for i in range(len(b_pred)):
        cut_pp.append(b_pred[i][cut_rr])
    
    return np.array(cut_pp)

def create_batches(feature, b_size, timesteps, feature_num=384):
    frms = np.ceil(len(feature) / timesteps)
    bss = np.ceil(frms / b_size).astype('int')
    
    pb = (feature_num-feature.shape[1]) // 2
    pt = feature_num-feature.shape[1]-pb
    l = len(feature)
    ch = feature.shape[2]
    pbb = np.zeros((l, pb, ch))
    ptt = np.zeros((l, pt, ch))
    feature = np.hstack([pbb, feature, ptt])

    BSS = []
    for i in range(bss):
        bs = np.zeros((b_size, timesteps, feature.shape[1], feature.shape[2]))
        for ii in range(b_size):
            start_i = i*b_size*timesteps + ii*timesteps
            if start_i >= len(feature):
                break
            end_i = min(start_i+timesteps, len(feature))
            length = end_i - start_i
            
            part = feature[start_i:start_i+length]
            bs[ii, 0:length] = part
        BSS.append(bs)
    
    return BSS

