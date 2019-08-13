
import h5py
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

from librosa import note_to_midi

def do_nothing():
    return

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

def infer_pitch_2(pitch):
    """
        Dim: time x 4 (off, dura, onset, offset)
    """
    
    # Threshold parameters for note finding
    ws = 16 # window size, ws*0.02sec
    bound = 2 # peak value detection for onset and offset, the more center, the more strict (value between 0~ws-1)
    occur_th = ws-2 # register
    shortest = 5 # shortest note duration
    offset_vs_dura = 2 # register a new note according to either offset(bigger val) or duration(smaller val) event

    pitch = np.insert(pitch, 0, np.zeros((ws, pitch.shape[1])), axis=0) # padding before the pitch occurence
    onset = []
    notes = []
    
    def register_note(on, end_t, kmsgs):
        # register one note
        if end_t-on >= shortest:
            nn = {"start": on,"end": end_t, **kmsgs}
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
                #nn = {"start": onset[0], "end": i+bound-1, "added": "onset"}
                #notes.append(nn)
                #del onset[0]
            
            onset.append(i+bound)
            continue
            
        if len(onset)==0:
            continue
        if np.sum(w_off[:occur_th]) > 0 and np.sum(w_off[occur_th:]) <= 0:
            register_note(onset[0], i+occur_th, {"added": "off_silence", "stren": pitch[onset[0], 2]})
    
    return notes
  
def infer_pitch(pitch):
    """
        Dim: time x 4 (off, dura, onset, offset)
    """
    
    # Threshold parameters for note finding
    ws = 16 # window size, ws*0.02sec
    bound = 7 # peak value detection for onset and offset, the more center, the more strict (value between 0~ws-1)
    occur_th = ws-2 # discard an onset event after this slience period of duration event 
    shortest = 5 # shortest note duration
    offset_vs_dura = 3 # register a new note according to either offset(bigger val) or duration(smaller val) event


    pitch = np.insert(pitch, 0, np.zeros((ws, pitch.shape[1])), axis=0) # padding before the pitch occurence
    onset = None
    dura = None
    notes = []
    
    def register_note(on, end_t, msg):
        # register one note
        if end_t-on >= shortest:
            nn = {"start": on,"end": end_t, "added": msg}
            notes.append(nn)
        
        onset = None
    
    for i in range(len(pitch)):
        window = pitch[i:i+ws]
        w_on = window[:,2]
        w_dur = window[:,1]
        w_off = window[:,3]
        
        if (w_on.argmax() == bound) and np.max(w_on)>0:
            # register onset occurence
            if onset is not None:
                # close previous onset and register a new note first
                nn = {"start": onset, "end": i+bound-1, "added": "onset"}
                notes.append(nn)
            
            onset = i+bound-1
            continue
        
        if onset is None:
            continue
        if np.sum(w_dur[:occur_th])<=0:
            if np.sum(w_off[:occur_th])<=0:
                if dura is not None:
                    # register a note according to last occurence of duration event
                    register_note(onset, dura, "dura")
                else:
                    register_note(onset, onset+shortest, "dura_none")
                onset = None
                
            elif w_off.argmax() == bound:
                # register one note
                end_t = i+bound-1
                register_note(onset, end_t, "offset")
                onset = None
            
        # P(offset | dura) 
        elif np.sum(w_dur[:occur_th])>0:
            if np.max(w_dur) > 0:
                dura = i+w_dur.argmax()-1
                
            if w_off.argmax() == bound and np.max(w_dur)<offset_vs_dura:
                # register one note
                end_t = i+bound-1
                register_note(onset, end_t, "off_under_dura")
                onset = None
    
    return notes
    
def infer_piece(piece):
    """
        Dim: time x 88 x 4 (off, dura, onset, offset)
    """
    assert(piece.shape[1] == 88), "Please down sample the pitch to 88 first (current: {}).format(piece.shape[1])"
    t_unit = 0.02 # constant, do not modify
    
    min_align_diff = 1 # to align the onset between notes with a short time difference 
    
    notes = []
    for i in range(88):
        print("Pitch: {}/{}".format(i+1, 88), end="\r")
        
        pitch = piece[:,i]
        if np.sum(pitch) <= 0:
            continue
            
        pns = infer_pitch_2(pitch)
        for ns in pns:
            ns["pitch"] = i
            notes.append(ns)
            
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
    v_map = lambda stren: int(s_low+((s_up-s_low)*((stren-l)/(u-l))))
    
    low_b = note_to_midi("A0")
    coll = set()
    for nn in notes:
        pitch = nn["pitch"] + low_b
        start = nn["start"] * t_unit
        end = nn["end"] * t_unit
        coll.add(v_map(nn["stren"]))
        m_note = pretty_midi.Note(velocity=v_map(nn["stren"]), pitch=pitch, start=start, end=end)
        piano.notes.append(m_note)
        
    midi.instruments.append(piano)
    return midi

def roll_down_sample(data, threshold=0.5, occur_num=2, base=88):
    total_roll = data.shape[1]
    assert total_roll % base == 0, "Wrong length: {}, {} % {} should be zero!".format(total_roll, total_roll, base)
    
    scale = round(total_roll/base)
    assert(occur_num>0 and occur_num<scale)
    
    return_v = np.zeros((len(data), base), dtype=int)
    
    for i in range(0, data.shape[1], scale):
        total = np.sum(data[:, i : i+scale], axis=1)
        return_v[:, int(i/scale)] = np.where(total>threshold*occur_num, total/occur_num, 0)
        
    return return_v
    
def down_sample(pred):
    dd = roll_down_sample(pred[:,:,0])
    for i in range(1, pred.shape[2]):
        dd = np.dstack([dd, roll_down_sample(pred[:,:,i])])

    return dd

def PostProcess(pred):
    onset = pred[:,:,2]
    dura = pred[:,:,1]
    
    #onset = np.where(onset<dura, 0, onset)
    
    # Normalize along each channel and filter by the nomalized value
    # onset channel
    onset = (onset-np.mean(onset))/np.std(onset)
    onset = np.where(onset<3, 0, onset)
    pred[:,:,2] = onset
    
    # duration channel
    dura = (dura-np.mean(dura))/np.std(dura)
    dura = np.where(dura<2, 0, dura)
    pred[:,:,1] = dura
    
    notes = infer_piece(down_sample(pred))
    midi = to_midi(notes)
    
    return midi
        
        
        
if __name__ == "__main__":
    f_name = "pred_angel_beats.hdf"
    f_name = "pred_onset_dura.hdf"
    #f_name = "pred_frame_overlap.hdf"
    f_name = "pred_onset_dura_sword.hdf"
    f_name = "pred_brave_song.hdf"
    f_name = "pred.hdf"
    p_in = h5py.File("HDF/"+f_name, "r")
    pred = p_in["0"][:]
    p_in.close()
    
    """
    f_name = "pred_frame_sword.hdf"
    p_in = h5py.File("HDF/"+f_name, "r")
    frm = p_in["0"][:]
    p_in.close()
    pred[:,:,1] = frm[:,:,1]
    """
    
    pp = pred#[16744:22000]
    midi = PostProcess(pp)
    midi.write("test.mid")
    
    draw_roll(midi)   
    plot3(pp)
        
        
        
