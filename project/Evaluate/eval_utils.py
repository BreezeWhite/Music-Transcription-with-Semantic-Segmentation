
import os
import h5py
import math
import librosa
import logging
import numpy as np

from project.utils import label_conversion
from project.central_frequency_352 import CentralFrequency


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

def full_label_conversion(label, timesteps):
    ll = []
    iters = math.ceil(len(label)/timesteps)
    for tid in range(iters):
        ll.append(label_conversion(
            label, 
            tid*timesteps,
            timesteps=timesteps,
            mpe=True
        ))

    return np.concatenate(ll)

def roll_down_sample(data, occur_num=3, base=88):
    # The input argument "data" should be thresholded 

    total_roll = data.shape[1]
    assert total_roll % base == 0, "Wrong length: {}, {} % {} should be zero!".format(total_roll, total_roll, base)
    
    scale = round(total_roll/base)
    assert(occur_num>0 and occur_num<scale)
    
    
    return_v = np.zeros((len(data), base), dtype=int)
    
    for i in range(0, data.shape[1], scale):
        total = np.sum(data[:, i : i+scale], axis=1)
        return_v[:, int(i/scale)] = np.where(total>=occur_num, total/occur_num, 0)
    return_v = np.where(return_v>1, 1, return_v)
        
    return return_v

def save_pred(preds, labels, out_path):
    ff = h5py.File(os.path.join(out_path, "pred.hdf"), "w")
    ll = h5py.File(os.path.join(out_path, "label.hdf"), "w")

    for i in range(len(preds)):
        ff.create_dataset(str(i), data=preds[i], compression="gzip", compression_opts=5)
        ll.create_dataset(str(i), data=labels[i], compression="gzip", compression_opts=5)

    ff.close()
    ll.close()

def find_occur(pitch, t_unit=0.02, min_duration=0.03):
    min_duration = max(t_unit, min_duration)
    
    candidate = np.where(pitch>0.5)[0]
    shifted   = np.insert(candidate, 0, 0)[:-1]
    
    diff   = candidate - shifted
    on_idx = np.where(diff>(min_duration/t_unit))[0]
    on_idx = candidate[on_idx]
    
    new_pitch = np.zeros_like(pitch)
    new_pitch[on_idx] = pitch[on_idx]
    onsets   = on_idx * t_unit
    interval = np.concatenate((onsets, onsets+2)).reshape(2, len(onsets)).transpose()
    
    return new_pitch, interval

def gen_onsets_info(data, t_unit=0.02):
    logging.debug("Data shape: %s", data.shape)
    pitches   = []
    intervals = []
    lowest_pitch = librosa.note_to_midi("A0")

    for i in range(data.shape[1]):
        _, it = find_occur(data[:, i], t_unit)
        
        if len(intervals)==0 and len(it) > 0:
            intervals = np.array(it)
        elif len(it) > 0:
            intervals = np.concatenate((intervals, np.array(it)), axis=0)
            
        # hz = CentralFrequency[i]
        hz = librosa.midi_to_hz(lowest_pitch+i)
        for i in range(len(it)):
            pitches.append(hz)
    
    if type(intervals) == list:
        intervals = np.array([]).reshape((0, 2))
    pitches = np.array(pitches)
    
    return intervals, pitches

def gen_frame_info(data, t_unit=0.02):
    t_idx, r_idx = np.where(data>0.5)
    #print("Length of estimated notes: ", len(t_idx))
    if len(t_idx) == 0:
        return np.array([]), []
    #f_idx = np.array(CentralFrequency)[r_idx]
    freq = [librosa.midi_to_hz(21+i) for i in range(88)]
    f_idx = np.array(freq)[r_idx]
    
    time_lst = []
    freq_lst = []
    t, uniq_t_idx = np.unique(t_idx, return_index=True)
    len_idx = len(uniq_t_idx)
    for i in range(len_idx-1):
        time_lst.append(t[i]*t_unit)
        
        f_lst = f_idx[uniq_t_idx[i] : uniq_t_idx[i+1]]
        freq_lst.append(np.array(f_lst))
    time_lst.append(t[-1]*t_unit)
    f_lst = f_idx[uniq_t_idx[-1]:]
    freq_lst.append(np.array(f_lst))

    return np.array(time_lst), freq_lst

def peak_picking(data, base=88):
    assert(len(data.shape)==2)
    assert(data.shape[1]%base == 0)

    scale = data.shape[1] // base
    new_data = np.zeros((len(data), base))

    for i in range(base):
        rr = range(i*scale, (i+1)*scale)
        new_data[:, i] = np.max(data[:, rr], axis=1)

    return new_data

