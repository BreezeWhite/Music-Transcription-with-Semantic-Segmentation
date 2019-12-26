

import os
import h5py
import librosa
import argparse
import numpy as np

import matplotlib.pyplot as plt



def ProcessLabel(gt_path, t_unit=0.02, length=None, pitch_width=352, base=88):
    """
        The variable 'length' should be the number of total frames.
    """

    with open(gt_path, "r") as ll_file:
        lines = ll_file.readlines()

    queue = []
    base_note = librosa.note_to_midi("A0")
    for i in range(1, len(lines)):
        onset, offset, midi = lines[i].split("\t")
        onset, offset, midi = float(onset), float(offset), int(midi[:midi.find("\n")])
        
        start_frm, end_frm = round(onset/t_unit), round(offset/t_unit)
        pitch = midi - base_note
        
        queue.append([start_frm, end_frm, pitch])
    
    if length is not None:
        assert(length >= queue[-1][1]), "The given length cannot be shorter than the real length! Please specify a longer one."
    else:
        length = queue[-1][1] + 100
    
    
    assert(pitch_width % base == 0)
    label = np.zeros((length, pitch_width))
    scale = pitch_width // base

    for note in queue:
        on, off, p = note
        p_range = range(p*scale, (p+1)*scale)
        
        label[on:off, p_range] = 1

    print("Time (sec): ", length*t_unit)
    
    
    return label

if __name__ == "__main__":
    ff_name = "./MapsDataset/ENSTDkCl/MUS/MAPS_MUS-bk_xmas4_ENSTDkCl.txt"
    
    label = ProcessLabel(ff_name)

    plt.imshow(label.transpose(), aspect='auto', origin='lower')
    plt.show()








    
    
    
    
