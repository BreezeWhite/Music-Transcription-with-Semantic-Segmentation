
import os
import h5py
import math
import librosa
import logging
import pretty_midi
import numpy as np

from project.utils import label_conversion
from project.configuration import MusicNetMIDIMapping
from project.central_frequency_352 import CentralFrequency

def norm(data):
    return (data-np.mean(data))/np.std(data)

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

def roll_down_sample(data, occur_num=2, base=88):
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

def find_occur(pitch, t_unit=0.02, min_duration=0.03):
    min_duration = max(t_unit, min_duration)
    min_frm = min_duration/t_unit

    cand = np.where(pitch>0.5)[0]
    if len(cand) == 0:
        return []

    start = cand[0]
    last = cand[0]
    note = []
    for cidx in cand:
        if cidx-last>1:
            if last-start>min_frm:
                note.append({"onset": start, "offset": last})
            start = cidx
        last = cidx
    
    return note

def gen_onsets_info_from_midi(midi, inst_num=1, t_unit=0.02):
    intervals = []
    pitches = []
    inst = midi.instruments[inst_num-1]
    for note in inst.notes:
        onset = note.start
        intervals.append([onset, onset+t_unit*2])
        pitches.append(librosa.midi_to_hz(note.pitch))

    return np.array(intervals), np.array(pitches)
    
def gen_onsets_info_from_label_v1(label, inst_num=1, t_unit=0.02):
    intervals = []
    pitches = []

    onsets = {}
    lowest_pitch = librosa.note_to_midi("A0")
    for t, ll in enumerate(label):
        for pitch, insts in ll.items():
            if inst_num not in insts:
                continue
            if (pitch not in onsets) or (insts[inst_num][0] > onsets[pitch]):
                intervals.append([t*t_unit, (t+2)*t_unit])
                pitches.append(librosa.midi_to_hz(lowest_pitch+pitch))
            onsets[pitch] = insts[inst_num][0]

    return np.array(intervals), np.array(pitches)

def gen_onsets_info_from_label(label, inst_num=1, t_unit=0.02):
    roll = label_conversion(label, 0, timesteps=len(label), onsets=True, mpe=True, ori_feature_size=88, feature_num=88)
    midi_ch_mapping = sorted([v for v in MusicNetMIDIMapping.values()])
    ch = midi_ch_mapping.index(inst_num)+1
    return gen_onsets_info(roll[:,:,ch], t_unit=t_unit)

def gen_onsets_info(data, t_unit=0.02):
    #logging.debug("Data shape: %s", data.shape)
    pitches   = []
    intervals = []
    lowest_pitch = librosa.note_to_midi("A0")

    for i in range(data.shape[1]):
        notes = find_occur(data[:, i], t_unit=t_unit)
        it = []
        for nn in notes:
            it.append([nn["onset"]*t_unit, (nn["onset"]+2)*t_unit])
        
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

def gen_frame_info_from_midi(midi, t_unit=0.02):
    inst = midi.instruments[0]
    tmp_midi = pretty_midi.PrettyMIDI()
    tmp_midi.instruments.append(inst)
    piano_roll = tmp_midi.get_piano_roll(fs=round(1/t_unit)).transpose()
    low = librosa.note_to_midi("A0")
    hi = librosa.note_to_midi("C8")
    piano_roll = piano_roll[:, low:hi]

    return gen_frame_info(piano_roll, t_unit=t_unit)

def gen_frame_info_from_label(label, inst_num=1, t_unit=0.02):
    roll = label_conversion(label, 0, timesteps=len(label), ori_feature_size=88, feature_num=88)
    midi_ch_mapping = sorted([v for v in MusicNetMIDIMapping.values()])
    ch = midi_ch_mapping.index(inst_num)+1
    return gen_frame_info(roll[:,:,ch], t_unit=t_unit)

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

