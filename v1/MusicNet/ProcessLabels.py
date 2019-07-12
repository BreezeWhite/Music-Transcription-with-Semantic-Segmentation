
import os
import csv
import pickle
import librosa
import numpy as np

from tqdm import trange


def get_csv_content(path):
    
    content = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        
        max_sample = 0
        for label in reader:
            start_time = int(label['start_time'])
            end_time   = int(label['end_time'])
            instrument = int(label['instrument'])
            note       = int(label['note'])
            start_beat = float(label['start_beat'])
            end_beat   = float(label['end_beat'])
            note_value = label['note_value']
            
            content.append([start_time, end_time, instrument, note, start_beat, end_beat, note_value])
            
            max_sample = max(max_sample, end_time)
            
    return content, max_sample

def process_labels(files, t_unit=0.01, sample_rate=44100):
    '''
    Unit of start_time and end_time is in sample (44.1 kHz)
    
    Midi number of instruments and their appearence rate in test/train(%):
        1  1  Piano        41.3  59.3
        2  7  Harpsichord   x     0.5
        3  41 Violin       19.3  16.3
        4  42 Viola         5.6   8.2
        5  43 Cello         9.5   9.4
        6  44 Contrabass    x     0.3
        7  61 Horn          5.3   1.3
        8  69 Oboe          x     0.7
        9  71 Bassoon       8.5   1.3
       10  72 Clarinet     10.4   2.1
       11  74 Flute         x     0.5
        
    Note pitch is in Midi number(0~127)
    
    Stored labels formated as follows:
        labels: total_pieces x total_frames x 88(piano roll) x instrument_list
         * Total 330 pieces in MusicNet dataset.
         * Label of each piece can be accessed by file name (string).
         * Each frame are <t_unit> long.
         * Number stored in <instrument_list> encode like above.
        
        i.e.
        pitch = labels['1759'][1000][pitch] (if that pitch is played at that frame, or exception would raise)
        You would get list of instrument numbers of that pitch ([1, 41], [61, 74, 7], ...)
    
    '''
    
    lowest_pitch = librosa.note_to_midi('A0')
    highest_pitch = librosa.note_to_midi('C8')
    pitches = highest_pitch - lowest_pitch + 1
        
    labels = []
    stat = {}
    contains = {}
    for idx in trange(len(files), leave=False):
        #print("{}/{}".format(idx+1, len(files)))
        item = files[idx]
        
        if not item.endswith('.csv'): continue
        
        content, max_sample = get_csv_content(item)
        
        frame_num = int(round(max_sample/sample_rate, 2)/t_unit)
        
        
        label = [{} for i in range(frame_num)]
        for cc in content:
            start_time, end_time, instrument, note, start_beat, end_beat, note_value = cc
            
            start_f, end_f = int(round(start_time/sample_rate, 2)/t_unit), int(round(end_time/sample_rate, 2)/t_unit)
            onsets_v = 1.0
            onsets_len = 2 # 2 frames long
            ii = 0
            for i in range(start_f, end_f):
                pitch = note-lowest_pitch
                if pitch not in label[i]:
                    label[i][pitch] = {}
                label[i][pitch][instrument] = onsets_v
                ii += 1
                if ii > onsets_len:
                    onsets_v /= (ii-onsets_len)
                
                
                # status
                if instrument not in stat:
                    stat[instrument] = 0
                stat[instrument] += 1
                
                # instrument is contained in pieces
                if instrument not in contains:
                    contains[instrument] = []
                item = item.split("/")[-1].split(".")[0]
                if item not in contains[instrument]:
                    contains[instrument].append(item)
                    contains[instrument].append(idx)
            
        labels.append(label)   
    
            
    return labels, stat, contains

def small_test(labels):
    count = 0
    total = 0
    for song in labels:
        total += len(song)
        for f_idx, frm in enumerate(song):
            for pitch in frm:
                if len(frm[pitch]) > 1:
                    count += 1
                    #print(song, f_idx, pitch, frm[pitch])
    print(count, total, count/total)    
        
def manage_label_process(path, num_per_file, save_path, t_unit=0.02):
    
    # Get the right index order according to the corressponding label folder
    target = path.split("_labels")[0] + "_data"
    files = os.listdir(target)
    files = [ff.split(".wav")[0] + ".csv" for ff in files]
    files = [os.path.join(path, ff) for ff in files]

    
    
    iters = np.ceil(len(files)/num_per_file)
    iters = int(iters)
    stats = {}
    contains = {}
    for i in trange(iters):
        sub_files = files[i*num_per_file : (i+1)*num_per_file]
        labels, stat, cntn = process_labels(sub_files, t_unit=t_unit)

        f_name = "train" if "train" in path else "test"
        post   = "_{}_{}_label.pickle".format(num_per_file, i+1)
        f_name += post
        f_name = os.path.join(save_path, f_name)

        pickle.dump(labels, open(f_name, 'wb'), pickle.HIGHEST_PROTOCOL)
        
        
        for key, val in stat.items():
            if key not in stats:
                stats[key] = 0
            stats[key] += val
            
        for key, vals in cntn.items():
            if key not in contains:
                contains[key] = []
                
            for j in range(0, len(vals), 2):
                p_idx = vals[j]
                p_pos = vals[j+1]
                if p_idx not in contains[key]:
                    contains[key].append(p_idx)
                    contains[key].append(p_pos)
                    contains[key].append(i+1)
                
            #for v in vals:
            #    if v not in contains[key]:
            #        contains[key].append(v)
        
    
    total = sum(stats.values())
    for k in stats:
        stats[k] /= total
    stats = sorted(stats.items(), key=lambda x:x[1], reverse=True)
    for kk, vv in stats:
        print("Midi number: ", kk, "Percentage: ", vv)
    
    #for key, vals in contains.items():
    #    with open("{}.txt".format(key), "w") as f:
    #        for i in range(0, len(vals), 3):
    #            f.write("{} {} {}\n".format(vals[i], vals[i+1], vals[i+2]))



if __name__ == "__main__":
    
    #labels = process_labels("../test_labels", t_unit=0.02)
    #t_ll   = process_labels("../train_labels", t_unit=0.02)
    #labels = dict(labels, **t_ll)
    #small_test(labels)
    
    save_path = "../features/20"
    
    #manage_process("../test_labels", 10, save_path, t_unit=0.02)
    manage_process("../train_labels", 20, save_path, t_unit=0.02)
    
    
    
    
        

