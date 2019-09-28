
import os
import csv
import h5py
import argparse
import mir_eval
import numpy as np

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Constant
MusicNet_Instruments = ["Piano", "Harpsichord", "Violin", "Viola", "Cello", "Contrabass", "Horn", "Oboe", "Bassoon", "Clarinet", "Flute"]


def sub_P_picking(data, base=88):
    
    assert(len(data.shape)==2)
    assert(data.shape[1]%base == 0)

    scale = data.shape[1] // base
    new_data = np.zeros((len(data), base))

    for i in range(base):
        rr = range(i*scale, (i+1)*scale)
        new_data[:, i] = np.max(data[:, rr], axis=1)

    return new_data

def peak_picking(data, base=88):
    if len(data.shape) == 2:
        return sub_P_picking(data, base)
    elif len(data.shape) == 3:
        
        for i in range(data.shape[2]):
            data[:, :base, i] = sub_P_picking(data[:,:,i], base)
        return data[:,:base]
    assert(False)

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

def process_onsets_occur(pred, ll, t_unit=0.02):
    # Dim: Time x roll_num
    
    new_pred = np.zeros_like(pred)
    new_ll   = np.zeros_like(ll)
    for i in range(pred.shape[1]):
        p, est_interval = find_occur2(pred[:, i])
        l, ref_interval = find_occur2(ll[:, i])
        
        match = []
        if len(est_interval) > 0 and len(ref_interval) > 0:
            match = mir_eval.transcription.match_note_onsets(ref_interval, est_interval)
            
        for m in match:
            pt = est_interval[m[1]][0]
            lt = ref_interval[m[0]][0]
            p_idx = int(round(pt/t_unit))
            l_idx = int(round(lt/t_unit))
            p[l_idx] = l[l_idx]
            p[p_idx] = 0
        
        new_pred[:, i] = p
        new_ll[:, i]   = l
        
    return new_pred, new_ll

 
def find_tp_fp_fn_idx(pred, label):
    idx_sm = np.where(label<0.5)
    idx_bg = np.where(label>0.5)
    flat_pred_sm = np.where(pred[idx_sm[0], idx_sm[1]] > 0.5)
    flat_pred_bg = np.where(pred[idx_bg[0], idx_bg[1]] < 0.5)
    flat_pred_t  = np.where(pred[idx_bg[0], idx_bg[1]] > 0.5)
    fp_idx = [idx_sm[0][flat_pred_sm[0]], idx_sm[1][flat_pred_sm[0]]]
    fn_idx = [idx_bg[0][flat_pred_bg[0]], idx_bg[1][flat_pred_bg[0]]]
    tp_idx = [idx_bg[0][flat_pred_t[0]], idx_bg[1][flat_pred_t[0]]]
    
    return tp_idx, fp_idx, fn_idx

def quantization(pred, label, threshold):
    pred  = np.where(pred>threshold, 1, 0)
    label = np.where(label>threshold, 1, 0)

    tp_idx, fp_idx, fn_idx = find_tp_fp_fn_idx(pred, label)

    MAX_V = 256
    pred = np.where(pred==1, MAX_V, pred)
    pred[fp_idx[0], fp_idx[1]] = MAX_V*0.7 # green, false-positive
    pred[fn_idx[0], fn_idx[1]] = MAX_V*0.3 # red, false-negative

    pred  = MAX_V - pred
    label = 1 - label
    
    return pred, label

def PLOT(data, save_name, plot_range, titles=None, color_map="terrain"):
    plt.clf()
    
    fig, axes = plt.subplots(nrows=data.shape[2])
    if type(axes) != list:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.imshow(data[:,:,i].transpose(), aspect=20, origin='lower', cmap=color_map)
    
    
    interval_num = 5
    sec_per_frm = 0.02
    lower, upper = min(plot_range), max(plot_range)+1
    interval = (upper-lower) // (interval_num-1)
    label_idx = [i*interval for i in range(interval_num)]
    
    for idx, ax in enumerate(axes):
        if titles is not None:
            ax.set_title(titles[idx])#MusicNet_Instruments[spec_inst[idx]])
        
        ax.set_xticks(label_idx)
        ax.set_xticklabels([str(int(i*sec_per_frm+lower*sec_per_frm)) for i in label_idx])
        
        ax.set_yticks([0, 40, 80])
        ax.set_ylabel("pitch number")
        #ax.set_ymargin(4)
    
    
    plt.xlabel("t(s)")
    plt.subplots_adjust(hspace=0.5)
    
    plt.savefig(save_name+".png", dpi=250)
    plt.close()

def plot_figures(pred, label, save_name, 
                 plot_range=range(500, 1500),
                 spec_inst=None,
                 quantize=True,
                 threshold=0.35,
                 max_row=4):
    
    print("Length: {} {}".format(len(pred), len(label)))
    
    
    #if spec_inst is not None:
    #    if 1 in pred.shape:
    #        pred = pred.squeeze()
    #    if len(pred.shape) < 3:
    #        spec_inst = None
    
    
    ## Select print range
    if len(pred) < max(plot_range):
        plot_range = range(0, len(pred))
    pred  = pred[plot_range]
    label = label[plot_range]
    
    ## Wether to filter values with threshold and print out fp, tp with different colors.
    color_map = "terrain"
    if quantize:
        color_map = "gist_ncar"
        if type(threshold) != list:
            threshold = [threshold for i in range(len(spec_inst))]
        else:
            assert(len(threshold) == len(spec_inst))
            
        for i in range(len(spec_inst)):
            idx = spec_inst[i]
            pp = pred[:,:,idx]
            ll = label[:,:,idx]
            th = threshold[i]
            
            pred[:,:,idx], label[:,:,idx] = quantization(pp, ll, th)
    
    data = []
    for i in range(len(spec_inst)):
        idx = spec_inst[i]
        tmp = np.dstack((pred[:,:,idx], label[:,:,idx]))
        data.append(tmp)
    tmp = data[0]
    for i in range(1, len(data)):
        tmp = np.dstack((tmp, data[i]))
    data = tmp

    ## Pring out piano rolls batch by batch
    Round = np.ceil(data.shape[2]/max_row).astype('int')
    for i in range(Round):
        if i == Round-1:
            rr = range(i*max_row, data.shape[2])
        else:
            rr = range(i*max_row, (i+1)*max_row)
        PLOT(data[:,:,rr], save_name+"_{}".format(i), plot_range, color_map=color_map)
    

if __name__ == "__main__":
    
    ## Parameter settings
    
    parser = argparse.ArgumentParser(description="Print out the figures of predicted piano roll.")
    parser.add_argument("-p", "--pred-path",
                        help="Path to the directory of prediction files.",
                        type=str)
    parser.add_argument("-o", "--output-path",
                        help="Save path of the output figures (default: %(default)s)",
                        type=str, default="./figures")
    parser.add_argument("-f", "--fig-name",
                        help="Name of the output figures (default: %(default)s)",
                        type=str, default="PianoRoll")                    
    parser.add_argument("-i", "--spec-instrument",
                        help="Specify which instruments to print out (default: %(default)s, All)",
                        type=int, nargs="+", default=[-1]) # -1 means print all                    
    parser.add_argument("-q", "--quantize",
                        help="Wether to print the original output value or thresholded value",
                        action="store_true")
    parser.add_argument("-t", "--threshold",
                        help="Thresholds to each channel. If using --quantize flag, the length must be the same as \
                               --spec-instrument. If not given, the program will check the configuration file in the \
                               path. If there is no config file, the program will use the default thresholds.",
                        type=float, nargs="+", default=[0.5 for i in range(11)])
    args = parser.parse_args()                    

    
    
    
    ##  Parameter post-process   
    assert(args.pred_path is not None)
    
    if args.quantize:
        config_f = os.path.join(args.pred_path, "configuration.csv")
        if os.path.exists(config_f):
            with open(config_f) as config:
                reader = csv.DictReader(config)
                th = []
                for inst in args.spec_instrument:
                    i_name = MusicNet_Instruments[inst]
                    th.append(reader[i_name])
            args.threshold = th
        assert(len(args.spec_instrument) == len(args.threshold))
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    ## Load prediction files
    print("Loading predictions...")
    with h5py.File(os.path.join(args.pred_path, "pred.hdf"), "r") as pred_in:
        data = [pred_in[key][:] for key in pred_in]
    
    ll_path = os.path.join(args.pred_path, "label.hdf")
    if os.path.exists(ll_path):
        with h5py.File(ll_path, "r") as label_in:
            label = [label_in[key][:] for key in label_in]
    else:
        ## If there is no label files, the label would just be the copy of predictions.
        label = [dd for dd in data]
    
    num = len(data)
    test_onsets = False
    
    if -1 in args.spec_instrument:
        args.spec_instrument = [i for i in range(data[0].shape[2])] #[i for i in range(len(MusicNet_Instruments))]


    ## Start to plot figures
    for i in range(num):
        pred, ll = data[0], label[0]
        
        if test_onsets:
            ## Preserve for future use
            pred, ll = np.where(pred > threshold, 1, 0), np.where(ll > 0.5, 1, 0)
            pred, ll = process_onsets_occur(pred, ll)
        
        ## Down sample from 352 to 88 of second dimension
        pred = peak_picking(pred.squeeze())
        ll   = peak_picking(ll.squeeze())
        
        print("Plot figures")
        plot_figures(pred, ll, 
                     save_name = "{}/{}_{}".format(args.output_path,args.fig_name, i), 
                     quantize  = args.quantize, 
                     spec_inst = args.spec_instrument,
                     threshold = args.threshold)
        
        del data[0], label[0]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
