
import os
import pickle
import mir_eval
import numpy as np

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def PLOT(pred, label, save_name, plot_range, inst_name=None, color_map="terrain", save_path="figures"):
    plt.clf()
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(pred.transpose(), aspect='auto', origin='lower', cmap=color_map)
    ax[1].imshow(label.transpose(), aspect='auto', origin='lower', cmap="terrain")
    
    
    interval_num = 5
    sec_per_frm = 0.02
    lower, upper = min(plot_range), max(plot_range)+1
    interval = (upper-lower) // (interval_num-1)
    label_idx = [i*interval for i in range(interval_num)]
    print(label_idx, lower, upper)
    
    ax[0].set_xticks(label_idx)
    ax[0].set_xticklabels([str(int(i*sec_per_frm+lower*sec_per_frm)) for i in label_idx])
    
    ax[0].set_yticks([0, 40, 80])
    
    
    plt.xlabel("t(s)")
    plt.ylabel("pitch number")
    
    
    plt.xlabel("t(s)")
    plt.ylabel("pitch number")
    
    
    if inst_name is not None:
        full_name = save_name + "_" + str(inst_name)
    else:
        full_name = save_name

    plt.savefig('{}/{}.png'.format(save_path, full_name), box_inches='tight', dpi=250)
    plt.close()

def plot_figures(pred, label, save_name, 
                 plot_range=range(1000, 2000),
                 spec_inst=None,
                 quantize=True,
                 threshold=0.35):
    
    print("Length: {} {}".format(len(pred), len(label)))
    
    
    if spec_inst is not None:
        if 1 in pred.shape:
            pred = pred.squeeze()
        if len(pred.shape) < 3:
            spec_inst = None
    
    # Select print range
    if len(pred) < max(plot_range):
        plot_range = range(0, len(pred))
    pred  = pred[plot_range]
    label = label[plot_range]
    
    # Wether to filter values with threshold and print out fp, tp with different colors.
    color_map = "terrain"
    if quantize:
        color_map = "gist_ncar"
        
        pred  = np.where(pred>threshold, 1, 0)
        label = np.where(label>threshold, 1, 0)

        tp_idx, fp_idx, fn_idx = find_tp_fp_fn_idx(pred, label)

        MAX_V = 256
        pred = np.where(pred==1, MAX_V, pred)
        pred[fp_idx[0], fp_idx[1]] = MAX_V*0.7 # green, false-positive
        pred[fn_idx[0], fn_idx[1]] = MAX_V*0.3 # red, false-negative

        pred  = MAX_V - pred
        label = 1 - label


    if spec_inst is None:
        if len(pred.shape) == 3:
            pred = pred[:,:,0]
        if len(label.shape) == 3:
            label = label[:,:,0]
        PLOT(pred, label, save_name, plot_range, color_map=color_map)
    else:
        for i in range(len(spec_inst)): 
            idx = spec_inst[i]
            p = pred[:,:,idx]
            l = label[:,:,idx]
            PLOT(p, l, save_name, plot_range, spec_inst[i], color_map=color_map)
    

if __name__ == "__main__":
    
    # Parameter settings
    num = 60
    threshold = 0.34
    channels  = [0, 6]
    #channels  = [i for i in range(12)]
    spec_inst = [2]#["All"]
    test_path = "/media/whitebreeze/本機磁碟/MusicNet/predictions"
    test_path = "./predictions"
    test_onsets = False
    ori_value = False

    pred_file = "test_10_1_pred352_MPE_"
    fig_name = "CFP_MusicNet"
    
    #pred_file = "test_10_1_pred352_CFP_MAPS"
    #fig_name = "CFP_MAPS"




    # Parameter post-process
    pred_file += str(channels)+".pickle"
    data_path = os.path.join(test_path, pred_file)    
    if type(spec_inst) == list and "All" in spec_inst:
        spec_inst = ["Piano", "Harpsichord", "Violin", "Viola", "Cello", "Contrabass", "Horn", "Oboe", "Bassoon", "Clarinet", "Flute"]
    #fig_name = "onsets" if test_onsets else "roll"

    print("Loading predicitons...")
    data = pickle.load(open(data_path, 'rb'))
    label = [data[i+1] for i in range(0, len(data), 2)]
    data  = [data[i] for i in range(0, len(data), 2)]

    
    num   = min(num, len(data))
    idx   = np.arange(num)
    #idx   = [2, 3, 6, 9]
    data  = [data[i] for i in idx]
    label = [label[i] for i in idx]
    num = len(idx)
    


    # Start to plot figures
    for i in range(num):
        pred, ll = data[0], label[0]
        
        if test_onsets:
            pred, ll = np.where(pred > threshold, 1, 0), np.where(ll > 0.5, 1, 0)
            pred, ll = process_onsets_occur(pred, ll)
        
        print("Plot figure")
        #plot_figures(pred, ll, "{}_352_{}".format(i, fig_name), 
        #             spec_inst=spec_inst, 
        #             quantize=not ori_value, 
        #             threshold=threshold)
        
        pred = peak_picking(pred.squeeze())
        ll   = peak_picking(ll.squeeze())
        plot_figures(pred, ll, "{}_88_{}".format(fig_name, i), 
                     quantize=not ori_value, 
                     spec_inst=spec_inst,
                     threshold=threshold)
        
        del data[0], label[0]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
