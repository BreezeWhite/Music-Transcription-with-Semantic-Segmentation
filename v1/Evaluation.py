
import os
import h5py
import pickle
import mir_eval
import argparse
import numpy as np

from project.configuration import MusicNet_Instruments
from project.central_frequency_352 import CentralFrequency

def hand_crafeted_evaluate(pred, 
                           label,
                           threshold=0.35,
                           t_unit=0.02):
    
    pred  = pred.transpose()
    label = label.transpose()
    
    epsilon = 0.00000001
    tp, fp, fn = 0, 0, 0
    for i, pitch in enumerate(pred):
        label[i], ref_interval = find_occur(label[i], 0.5, t_unit)
        pred[i],  t_interval   = find_occur(pitch, threshold, t_unit)

        if len(t_interval) > 0 and len(ref_interval) > 0:
            match = mir_eval.transcription.match_note_onsets(ref_interval, t_interval)
            tp += len(match)
            fp += len(t_interval) - len(match)
            fn += len(ref_interval) - len(match)
        elif len(t_interval) > 0:
            fp += len(t_interval)
        elif len(ref_interval) > 0:
            fn += len(ref_interval)

    precision = tp/(tp+fp+epsilon)
    recall    = tp/(tp+fn+epsilon)
    accuracy  = tp/(tp+fp+fn+epsilon)
    fscore    = 2*precision*recall/(precision+recall+epsilon)
    
    return precision, recall, accuracy, fscore

def find_occur2(pitch, threshold=0.35, t_unit=0.02, min_duration=0.03):
    min_duration = max(t_unit, min_duration)
    
    candidate = np.where(pitch>threshold)[0]
    shifted   = np.insert(candidate, 0, 0)[:-1]
    
    diff   = candidate - shifted
    on_idx = np.where(diff>(min_duration/t_unit))[0]
    on_idx = candidate[on_idx]
    
    new_pitch = np.zeros_like(pitch)
    new_pitch[on_idx] = pitch[on_idx]
    onsets   = on_idx * t_unit
    interval = np.concatenate((onsets, onsets+2)).reshape(2, len(onsets)).transpose()
    
    return new_pitch, interval
    

def find_occur(pitch, threshold=0.35, t_unit=0.02):
    pitch_on = False
    new_pitch = np.zeros_like(pitch)
    t_interval = []
    
    for i, p in enumerate(pitch):
        if not pitch_on:
            if p > threshold:
                pitch_on = True
                new_pitch[i] = p
                t_interval.append([i*t_unit, i*t_unit+2])
        else:
            if p<=threshold:
                pitch_on = False
                
    return new_pitch, np.array(t_interval)


def gen_onsets_info(data,
                    threshold=0.35,
                    t_unit=0.02):
    
    pitches   = []
    intervals = []
    
    for i in range(data.shape[1]):
        _, it = find_occur2(data[:, i], threshold, t_unit)
        
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

def gen_frame_info(data,
                   threshold=0.35,
                   t_unit=0.02):

    t_idx, r_idx = np.where(data>threshold)
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

def evaluate(preds, 
             labels,
             t_unit=0.02,
             threshold=0.35,
             onsets=True):
    
    prec, rec, acc, f = 0, 0, 0, 0
    for i in range(len(preds)):
        print("({}/{})".format(i+1, len(preds)))
        
        # Down sample from 352 to 88
        preds[i]  = peak_picking(preds[i]) if preds[i].shape[1] > 88 else preds[i]
        labels[i] = peak_picking(labels[i]) if labels[i].shape[1] > 88 else labels[i]
        
        
        accuracy = 0
        if onsets:
            est_interval, est_hz = gen_onsets_info(preds[i],  threshold, t_unit=t_unit)
            ref_interval, ref_hz = gen_onsets_info(labels[i], threshold, t_unit=t_unit)
            out = mir_eval.transcription.precision_recall_f1_overlap(ref_interval, ref_hz, 
                                                                     est_interval, est_hz, 
                                                                     offset_ratio=None)
            precision, recall, fscore, avg_overlap_ratio = out
        else:
            # Some hack 
            #pred = preds[i][:,:,1]
            #pred = np.where(pred>preds[i][:,:,0], 1, 0)
            #preds[i] = pred
            # End hack

            est_time, est_hz = gen_frame_info(preds[i], threshold, t_unit) 
            ref_time, ref_hz = gen_frame_info(labels[i], threshold, t_unit)
            out = mir_eval.multipitch.metrics(ref_time, ref_hz, est_time, est_hz)

            precision, recall, accuracy = out[0:3]
            prec_chroma, rec_chroma, acc_chroma = out[7:10]

            fscore = 2*precision*recall/(precision+recall+1e-8)

        print("Prec: {:.4f} Rec: {:.4f} Acc: {:.4f} F-score: {:.4f}".format(precision, recall, accuracy, fscore))
        
        prec += precision
        rec  += recall
        acc  += accuracy
        f    += fscore
    
    num = len(preds)
    print("\nTotal average: ")
    print("Prec: {:.4f} Rec: {:.4f} Acc: {:.4f} F-score: {:.4f}".format(prec/num, rec/num, acc/num, f/num))
    
    return f/num

def merge(data):
    if len(data.shape) == 3:
        return np.nanmax(data, axis=2)
    return data

def threshold_finding(preds, labels, onsets=False, mix=False, spec_inst=0):
    if mix:
        preds = [merge(p) for p in preds]
        labels = [merge(l) for l in labels]
    else:
        preds = [p[:,:,spec_inst] for p in preds]
        labels = [l[:,:,spec_inst] for l in labels]
    MAX = max([np.max(p) for p in preds])
    

    th = [i/1000 for i in range(round(MAX*0.4*1000).astype('int'), round(MAX*0.9*1000).astype('int'), 5)]

    best = {'f': 0, 'th': -1}
    print(len(th), th)
    for idx, t in enumerate(th):
        print("{}/{}".format(idx+1, len(th)))
        print("Threshold: ", t)
        f = evaluate(preds, labels, threshold=t, onsets=onsets)
        if f > best['f']:
            best['f'] = f
            best['th'] = t

    return best, th

def load_prediction(path, pred_name="pred.hdf", label_name="label.hdf"):
    pred_file = os.path.join(path, pred_name)
    label_file = os.path.join(path, label_name)
    assert(os.path.exists(pred_file) and os.path.exists(label_file))


    pred_in = h5py.File(pred_file, "r")
    label_in = h5py.File(label_file, "r")
    
    pred = []
    label = []
    for key in pred_in:
        pred.append(pred_in[key][:])
        label.append(label_in[key][:])
    pred_in.close()
    label_in.close()

    return pred, label

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate on the predictions")
    parser.add_argument("--val-pred-path",
                        help="Path to directory of prediction files of validation set. If not given, \
                              the threshold used during the evaluation will be directly set based on \
                              the test set.",
                        type=str)
    parser.add_argument("--test-pred-path",
                        help="Path to directory of prediction files of test set.",
                        type=str)
    parser.add_argument("--spec-instrument",
                        help="To evaluate on the specific channel. This option is for MusicNet. \
                              (default %(default)d)",
                        type=int, default=0)
    parser.add_argument("--merge-channels",
                        help="Merge channels into single channel. Keep the maximum value among \
                             all channels. If this parameter is given, the parameter, --spec-instrument, \
                             can be ignored.",
                        action="store_true")
    
    args = parser.parse_args()

    if args.val_pred_path is None:
        args.val_pred_path = args.test_pred_path

    # Parameter settings
    
    validate = False
    onsets = False
    
    
    best = {"th": 0.10}

    # Validation
    if validate:
        print("Loading validation predictions: ", args.val_pred_path)
        v_preds, v_labels = load_prediction(args.val_pred_path)
        best, thresholds = threshold_finding(v_preds, v_labels, onsets, args.merge_channels, args.spec_instrument)
        print("Best setting: ", best, "Searched thresholds: ", thresholds)

        del v_preds, v_labels

    # Test
    if args.val_pred_path is None:
        t_preds, t_labels = v_preds, v_labels
    else:
        print("Loading testing predictions: ", args.test_pred_path)
        t_preds, t_labels = load_prediction(args.test_pred_path)
    
    if args.merge_channels:
        t_preds = [merge(p) for p in t_preds]
        t_labels = [merge(l) for l in t_labels]
    else:
        t_preds = [p[:,:,args.spec_instrument+1] for p in t_preds]
        t_labels = [l[:,:,args.spec_instrument+1] for l in t_labels]

    evaluate(t_preds, t_labels, threshold=best['th'], onsets=onsets)
    print("Result of evalution on " + MusicNet_Instruments[args.spec_instrument])
    print("Best setting: ", best, "Searched thresholds: ", thresholds)

            
            
            
    
    
