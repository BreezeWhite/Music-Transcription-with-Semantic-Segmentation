import sys
sys.path.append("./")

import h5py
import math
import pretty_midi
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from librosa import note_to_midi
from project.configuration import MusicNet_Instruments, MusicNetMIDIMapping
from project.Evaluate.eval_utils import roll_down_sample, find_occur

def plot3(pred):
    fig, axes = plt.subplots(nrows=2)
    
    th = 0.5
    
    on = pred[:,:,1]
    on = (on-np.mean(on))/np.std(on)
    #on_th = np.where(on>2.5, 1, 0)
    on[on<th] = 0
    on = roll_down_sample(on)
    axes[0].imshow(on.transpose(), origin="lower", aspect="auto")
    
    on = pred[:,:,2]
    on = (on-np.mean(on))/np.std(on)
    #on_th = np.where(on>5, 1, 0)
    on[on<th] = 0
    on = roll_down_sample(on)
    axes[1].imshow(on.transpose(), origin="lower", aspect="auto")
    
    """
    on = pred[:,:,3]
    on = (on-np.mean(on))/np.std(on)
    #axes[2].imshow(on.transpose(), origin="lower", aspect="auto")
    #on_th = np.where(on>10, 1, 0)
    on[on<th] = 0
    on = roll_down_sample(on)
    axes[2].imshow(on.transpose(), origin="lower", aspect="auto")
    """
    plt.show()
    
def draw_roll(mm):
    roll = mm.get_piano_roll(fs=50)
    roll = np.where(roll>0, 1, 0)
    plt.imshow(roll, aspect="auto", origin="lower")
    
    plt.show()

def infer_pitch_v2(pitch, shortest=5, offset_interval=6):
    w_on = pitch[:,2]
    w_dura = pitch[:,1]

    peaks, properties = find_peaks(w_on, distance=shortest, width=5)
    if len(peaks) == 0:
        return []

    notes = []
    adjust = 5 if shortest==10 else 2
    for i in range(len(peaks)-1):
        notes.append({"start": peaks[i]-adjust, "end": peaks[i+1]-adjust, "stren": pitch[peaks[i], 2]})
    notes.append({"start": peaks[-1]-adjust, "end": len(w_on), "stren": pitch[peaks[-1], 2]})

    del_idx = []
    for idx, p in enumerate(peaks):
        upper = int(peaks[idx+1]) if idx<len(peaks)-2 else len(w_dura)
        for i in range(p, upper):
            if np.sum(w_dura[i:i+offset_interval]) == 0:
                if i - notes[idx]["start"] < shortest:
                    del_idx.append(idx)
                else:
                    notes[idx]["end"] = i
                break

    for ii, i in enumerate(del_idx):
        del notes[i-ii]

    return notes
    
def infer_pitch(pitch):
    """
        Dim: time x 4 (off, dura, onset, offset)
    """
    
    # Threshold parameters for note finding
    ws = 12 # window size, ws*0.02sec
    bound = ws//2 # peak value detection for onset and offset, the more center, the more strict (value between 0~ws-1)
    occur_th = ws//2 # register
    shortest = 5 # shortest note duration
    offset_vs_dura = 2 # register a new note according to either offset(bigger val) or duration(smaller val) event

    pitch = np.insert(pitch, 0, np.zeros((ws, pitch.shape[1])), axis=0) # padding before the pitch occurence
    onset = []
    notes = []
    
    def register_note(on, end_t, kmsgs):
        # register one note
        if end_t-on >= shortest:
            nn = {"start": on-bound, "end": end_t-bound, **kmsgs}
            notes.append(nn)
        
        del onset[0]
    
    for i in range(len(pitch)):
        window = pitch[i:i+ws]
        w_on = window[:,2]
        w_off = window[:,1] + window[:,2] #+ window[:,3] # onset + duration + offset
        
        if (w_on.argmax() == bound) and np.max(w_on)>0:
            # register onset occurence
            if len(onset)>0:
                # close previous onset and register a new note first
                register_note(onset[0], i+bound, {"added": "onset", "stren": pitch[onset[0], 2]})
            onset.append(i)
            continue
            
        if len(onset)==0:
            continue
        if np.sum(w_off[:occur_th]) > 0 and np.sum(w_off[occur_th:]) <= 0:
            register_note(onset[0], i+occur_th, {"added": "off_silence", "stren": pitch[onset[0], 2]})
    
    return notes
  
def infer_piece(piece, shortest_sec=0.1, offset_sec=0.12, t_unit=0.02):
    """
        Dim: time x 88 x 4 (off, dura, onset, offset)
    """
    assert(piece.shape[1] == 88), "Please down sample the pitch to 88 first (current: {}).format(piece.shape[1])"
    min_align_diff = 1 # to align the onset between notes with a short time difference 
    
    notes = []
    for i in range(88):
        print("Pitch: {}/{}".format(i+1, 88), end="\r")
        
        pitch = piece[:,i]
        if np.sum(pitch) <= 0:
            continue
            
        #pns = infer_pitch(pitch)
        pns = infer_pitch_v2(pitch, shortest=round(shortest_sec/t_unit), offset_interval=round(offset_sec/t_unit))
        for ns in pns:
            ns["pitch"] = i
            notes.append(ns)
    print(" "*80, end="\r")        

    notes = sorted(notes, key=lambda d: d["start"])
    last_start = 0
    for i in range(len(notes)):
        start_diff = notes[i]["start"] - last_start 
        if start_diff < min_align_diff:
            notes[i]["start"] -= start_diff
            notes[i]["end"] -= start_diff
        else:
            last_start = notes[i]["start"]
            
    return notes
    
def find_min_max_stren(notes):
    MIN = 999
    MAX = 0
    for nn in notes:
        nn_s = nn["stren"]
        if nn_s > MAX:
            MAX = nn_s
        if nn_s < MIN:
            MIN = nn_s
    
    return MIN, MAX

def to_midi(notes, t_unit=0.02):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    
    l, u = find_min_max_stren(notes)
    s_low = 60
    s_up = 127
    v_map = lambda stren: int(s_low+((s_up-s_low)*((stren-l)/(u-l+0.0001))))
    
    low_b = note_to_midi("A0")
    coll = set()
    for nn in notes:
        pitch = nn["pitch"] + low_b
        start = nn["start"] * t_unit
        end = nn["end"] * t_unit
        volume = v_map(nn["stren"])
        coll.add(pitch)
        m_note = pretty_midi.Note(velocity=volume, pitch=pitch, start=start, end=end)
        piano.notes.append(m_note)
    midi.instruments.append(piano)
    return midi

def down_sample(pred):
    dd = roll_down_sample(pred[:,:,0])
    for i in range(1, pred.shape[2]):
        dd = np.dstack([dd, roll_down_sample(pred[:,:,i], occur_num=3)])

    return dd

def interpolation(data, ori_t_unit=0.02, tar_t_unit=0.01):
    assert(len(data.shape)==2)

    ori_x = np.arange(len(data))
    tar_x = np.arange(0, len(data), tar_t_unit/ori_t_unit)
    func = CubicSpline(ori_x, data, axis=0)
    return func(tar_x)

def norm(data):
    return (data-np.mean(data))/np.std(data)

def norm_onset_dura(pred, onset_th, dura_th, interpolate=True):
    length = len(pred)*2 if interpolate else len(pred)
    norm_pred = np.zeros((length,)+pred.shape[1:])
    onset = interpolation(pred[:,:,2])
    dura = interpolation(pred[:,:,1])
    
    onset = np.where(onset<dura, 0, onset)
    norm_onset = norm(onset)
    onset = np.where(norm_onset<onset_th, 0, norm_onset)
    norm_pred[:,:,2] = onset

    norm_dura = norm(dura)+onset
    dura = np.where(norm_dura<dura_th, 0, norm_dura)
    norm_pred[:,:,1] = dura

    return norm_pred

def norm_split_onset_dura(pred, onset_th, lower_onset_th, split_bound, dura_th, interpolate=True):
    upper_range = range(4*split_bound, 352)
    upper_pred = pred[:,upper_range]
    upper_pred = norm_onset_dura(upper_pred, onset_th, dura_th, interpolate)

    lower_range = range(4*split_bound)
    lower_pred = pred[:,lower_range]
    lower_pred = norm_onset_dura(lower_pred, lower_onset_th, dura_th, interpolate)

    return np.hstack([lower_pred, upper_pred])

def draw(data, out_name="roll.png"):
    plt.imshow(data.transpose(), origin="lower", aspect="auto")
    plt.savefig(out_name, dpi=250)

def PostProcess(pred, 
                mode="note", 
                onset_th=7.5, 
                lower_onset_th=None,
                split_bound=36,
                dura_th=2, 
                frm_th=1, 
                t_unit=0.02):
    if mode=="note" or mode=="mpe_note":
        if lower_onset_th is not None:
            norm_pred = norm_split_onset_dura(pred, onset_th=onset_th, lower_onset_th=lower_onset_th, split_bound=split_bound,
                                              dura_th=dura_th, interpolate=True)
        else:
            norm_pred = norm_onset_dura(pred, onset_th=onset_th, dura_th=dura_th, interpolate=True)

        norm_pred = np.where(norm_pred>0, norm_pred+1, 0)
        notes = infer_piece(down_sample(norm_pred), t_unit=0.01)
        midi = to_midi(notes, t_unit=t_unit/2)
    
    elif mode=="frame" or mode=="mpe_frame":
        ch_num = pred.shape[2]
        if ch_num == 2:
            mix = pred[:,:,1]
        elif ch_num == 3:
            mix = (pred[:,:,1] + pred[:,:,2])/2
        else:
            raise ValueError("Unknown channel length: {}".format(ch_num))
        
        p = norm(mix) 
        p = np.where(p>frm_th, 1, 0)
        p = roll_down_sample(p)
        
        notes = []
        for idx in range(p.shape[1]):
            p_note = find_occur(p[:,idx], t_unit=t_unit)
            for nn in p_note:
                note = {
                    "pitch": idx,
                    "start": nn["onset"],
                    "end": nn["offset"],
                    "stren": mix[int(nn["onset"]*t_unit), idx*4]
                }
                notes.append(note)
        midi = to_midi(notes, t_unit=t_unit)

    else:
        raise ValueError("Supported mode are ['note', 'frame']. Given mode: {}".format(mode))

    return midi
        
def threshold_type_converter(th, length):
    if isinstance(th, list):
        assert(len(th)==length)
    else:
        th = [th for _ in range(length)]
    return th

def entropy(data, bins=200):
    min_v = -20#np.min(data)
    max_v = 30#np.max(data)
    interval = (max_v-min_v)/bins
    cut_offs = [min_v+i*interval for i in range(bins+1)]
    discrete_v = np.digitize(data, cut_offs)
    _, counts = np.unique(discrete_v, return_counts=True)
    probs = counts/np.sum(counts)
    ent = 0
    for p in probs:
        ent -= p * math.log(p, math.e)

    return ent

def MultiPostProcess(pred, mode='note', onset_th=5, dura_th=2, frm_th=1, inst_th=0.95, t_unit=0.02):
    """ Function for post-process multi-instrument prediction
    Parameters:
        mode: 'note' or 'frame'
        onset_th: Threshold of onset channel. Type of list or float
        dura_th: Threshold of duration channel. Type of list or float
        inst_th: Threshold of deciding a instrument is present or not according to Std. of prediction.
    """
    if mode=='note' or mode=='mpe_note':
        ch_per_inst = 2
    elif mode=='frame' or mode=='mpe_frame':
        ch_per_inst = 1
        ch_per_inst = 2 # frame-level hack
    elif mode=='offset':
        raise NotImplementedError
    else:
        raise ValueError
    assert((pred.shape[-1]-1)%ch_per_inst == 0), f"Input shape: {pred.shape}"
    
    ch_container = []
    iters = (pred.shape[-1]-1)//ch_per_inst
    for i in range(ch_per_inst):
        # First item would be duration channel
        # Second item would be onset channel
        # Third item would be offset channel (not yet implement)
        item = pred[:,:,[it*ch_per_inst+i+1 for it in range(iters)]]
        ch_container.append(norm(item))

    if mode.startswith("mpe_"):
        # Some different process for none-instrument care cases
        # Merge all channels into first channel
        iters = 1
        chs = ch_container[0].shape[-1]
        for i in range(ch_per_inst):
            pp = ch_container[i]
            pp[:,:,0] = np.average(pp, axis=2)
            ch_container[i] = pp

    onset_th = threshold_type_converter(onset_th, iters)
    dura_th = threshold_type_converter(dura_th, iters)
    frm_th = threshold_type_converter(frm_th, iters)

    zeros = np.zeros((pred.shape[:-1]))
    midis = []
    note_num = []
    out_midi = pretty_midi.PrettyMIDI()
    for i in range(iters):
        normed_ch = []
        std = 0
        ent = 0
        for ii in range(ch_per_inst):
            ch = ch_container[ii][:,:,i]
            std += np.std(ch)
            ent += entropy(ch)
            normed_ch.append(ch)
        print("std: {:.3f} ent: {:.3f} mult: {:.3f}".format(std/ch_per_inst, ent/ch_per_inst, std*ent/ch_per_inst**2))
        if iters>1 and (std/ch_per_inst < inst_th):
            continue

        pp = np.dstack([zeros] + normed_ch)
        midi = PostProcess(pp, mode=mode, onset_th=onset_th[i], dura_th=dura_th[i], frm_th=frm_th[i], t_unit=t_unit)

        inst_name = MusicNet_Instruments[i]
        program = MusicNetMIDIMapping[inst_name]
        inst = pretty_midi.Instrument(program=program, name=inst_name)
        inst.notes = midi.instruments[0].notes
        out_midi.instruments.append(inst)

    return out_midi

        
if __name__ == "__main__":
    f_name = "./prediction/musicnet_multi_note_prediction/MusicNet-Attn-Note-W4.2_predictions.hdf"
    p_in = h5py.File(f_name, "r")
    keys = list(p_in.keys())
    print(keys)
    pred = p_in["2298"][:]
    p_in.close()

    for i in range(11):
        plt.imshow(pred[:2000,:,i*2+1].transpose(), origin="lower", aspect="auto")
        #plt.savefig("{}_frm.png".format(i), dpi=250)
        plt.imshow(pred[:2000,:,i*2+2].transpose(), origin="lower", aspect="auto")
        #plt.savefig("{}_onset.png".format(i), dpi=250)
    
    # midi = PostProcess(pred, mode="frame")
    onset_th = [2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    midi = MultiPostProcess(pred, onset_th=5)#onset_th)
    midi.write("test.mid")
    
    #draw_roll(midi)   
    #plot3(pp)
        
        
        
