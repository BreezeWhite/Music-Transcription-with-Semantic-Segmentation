
import os
import h5py
import numpy as np

from project.central_frequency_352 import CentralFrequency


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
    
    pitches   = []
    intervals = []
    
    for i in range(data.shape[1]):
        _, it = find_occur(data[:, i], t_unit)
        
        if len(intervals)==0 and len(it) > 0:
            intervals = np.array(it)
        elif len(it) > 0:
            intervals = np.concatenate((intervals, np.array(it)), axis=0)
            
        hz = CentralFrequency[i]
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
    f_idx = np.array(CentralFrequency)[r_idx]
    
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

