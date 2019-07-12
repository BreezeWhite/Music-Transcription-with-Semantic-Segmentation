
import pretty_midi
import numpy as np

from librosa import note_to_midi
from project.Evaluate.eval_utils import roll_down_sample

def do_nothing():
    return

def infer_pitch(pitch):
    """
        Dim: time x 4 (off, dura, onset, offset)
    """
    
    # Threshold parameters for note finding
    ws = 8 # window size, 8*0.02sec = 0.16sec
    bound = 3 # peak value detection, the more center, the more strict (value between 0~ws-1)
    occur_th = ws-1 # discard an onset event after this slience period of duration event 
    shortest = 5 # shortest note duration
    offset_vs_dura = 1 # register a new note according to either offset(bigger val) or duration(smaller val) event


    pitch = np.insert(pitch, 0, np.zeros((ws, pitch.shape[1])), axis=0) # padding before the pitch occurence
    onset = None
    dura = None
    notes = []
    for i in range(len(pitch)):
        window = pitch[i:i+ws]
        w_on = window[:,2]
        w_dur = window[:,1]
        w_off = window[:,3]
        
        dura = i if pitch[i, 1] > 0 else dura
        
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
                if dura is not None and dura-onset >= shortest:
                    # register a note according to last occurence of duration event
                    nn = {"start": onset, "end": dura, "added": "dura"}
                    notes.append(nn)
                
                onset = None

            elif w_off.argmax() == bound:
                # register one note
                end_t = i+bound-1
                if end_t-onset < shortest:
                    end_t = onset+shortest
                    
                nn = {"start": onset,"end": end_t, "added": "offset"}
                notes.append(nn)
                
                onset = None
            
        # P(offset | dura) 
        elif np.sum(w_dur[:occur_th])>0:
            if w_off.argmax() == bound and np.max(w_dur)<offset_vs_dura:
                # register one note
                end_t = i+bound-1
                if end_t-onset < shortest:
                    end_t = onset+shortest
                    
                nn = {"start": onset,"end": end_t, "added": "off_under_dura"}
                notes.append(nn)
                
                onset = None
    
    return notes
    
def infer_piece(piece):
    """
        Dim: time x 88 x 4 (off, dura, onset, offset)
    """
    assert(piece.shape[1] == 88), "Please down sample the pitch to 88 first (current: {}).format(piece.shape[1])"
    t_unit = 0.02 # constant, do not modify
    
    min_align_diff = 4 # to align the onset between notes with a short time difference 
    
    notes = []
    for i in range(88):
        print("Pitch: {}/{}".format(i+1, 88))
        
        pitch = piece[:,i]
        if np.sum(pitch) <= 0:
            continue
            
        pns = infer_pitch(pitch)
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
    
def to_midi(notes, t_unit=0.02):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    low_b = note_to_midi("A0")
    for nn in notes:
        pitch = nn["pitch"] + low_b
        start = nn["start"] * t_unit
        end = nn["end"] * t_unit
        m_note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        piano.notes.append(m_note)
        
    midi.instruments.append(piano)
    
    return midi

def down_sample(pred):
    dd = roll_down_sample(pred[:,:,0])
    for i in range(1, pred.shape[2]):
        dd = np.dstack([dd, roll_down_sample(pred[:,:,i])])

    return dd

def PostProcess(pred):    
    
    # Normalize along each channel and filter by the nomalized value
    # onset channel
    ch = pred[:,:,2]
    ch = (ch-np.mean(ch))/np.std(ch)
    ch[ch<3.2] = 0
    pred[:,:,2] = ch

    # offset channel
    ch = pred[:,:,3]
    ch = (ch-np.mean(ch))/np.std(ch)
    ch[ch<1.5] = 0
    pred[:,:,3] = ch

    # duration channel
    ch = pred[:,:,1]
    ch = (ch-np.mean(ch))/np.std(ch)
    ch[ch<0.5] = 0
    pred[:,:,1] = ch

    notes = infer_piece(down_sample(pred))
    midi = to_midi(notes)
    
    return notes, midi
        
        
        
        
        
        
        
