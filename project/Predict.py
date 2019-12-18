import numpy as np
from scipy.special import expit

def predict(feature, model, timesteps, feature_num=384, batch_size=4, overlap_ratio=0.5):
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
    pred_buffer = np.zeros_like(feature)
    
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
        
    return expit(pred_buffer[pad_begin_len:pad_begin_len+length, pb:feature_num-pt] * overlap_ratio)

