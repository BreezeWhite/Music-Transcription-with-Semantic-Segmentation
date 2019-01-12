import os
import ast
import csv
import h5py
import pickle
import numpy as np

from scipy.special import expit
from project.configuration import get_MusicNet_label_num_mapping, get_instruments_num, MusicNet_Instruments
from keras.models import model_from_json, model_from_yaml
from mir_eval import melody


TIMESTEP = 128
SUBDIVISION = 8


def freq2midi(f):
    return 69 + 12*np.log2(f/440)


def midi2freq(m):
    return 2**((m - 69)/ 12) * 440


def sigmoid(x):
    #return 1 / (1 + np.exp(-x))
    return expit(x)

def note_res_downsampling(score):
    # filter
    f = [0.1, 0.2, 0.4, 0.2, 0.1]
    r = len(f) // 2
    
    shape = (score.shape[2],) if len(score.shape)==3 else None
    
    new_score_shape = (score.shape[0], 88) + shape if shape is not None else (score.shape[0], 88)
    new_score = np.zeros(new_score_shape)
    
    pad_shape = (new_score.shape[0], 2) + shape if shape is not None else (new_score.shape[0], 2)
    pad = np.zeros(pad_shape)
    score = np.concatenate([pad, score], axis=1)
    
    f_aug_shape = (new_score.shape[0], 1) + shape if shape is not None else (new_score.shape[0], 1)
    f_aug = np.tile(f, f_aug_shape)

    for i in range(0, 352, 4):
        cent = i + 2
        lb = max(0, cent - r)
        ub = min(353, (cent + 1) + r)
        rr = np.sum(score[:, lb:ub] * f_aug, axis=1)
        new_score[:, i // 4] = rr
    return new_score

  

def padding(x,
            feature_num,
            timesteps,
            dimension=False,
            muti_instrument=False):
    
    if muti_instrument:
        new_x = [{} for _ in range(timesteps)]
        for p in x:
            new_x.append(p)
        for _ in range(timesteps):
            new_x.append({})
        
        return new_x



    extended_chorale = np.array(x)

    if (((feature_num - x.shape[1]) % 2) == 0):
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t

    else:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t + 1
    
    
    ext_dim = (extended_chorale.shape[2],) if len(extended_chorale.shape)==3 else None
    
    top = np.zeros((extended_chorale.shape[0], p_t)+ext_dim if ext_dim is not None else (extended_chorale.shape[0], p_t))
    bottom = np.zeros((extended_chorale.shape[0], p_b)+ext_dim if ext_dim is not None else (extended_chorale.shape[0], p_b))
    extended_chorale = np.concatenate([top, extended_chorale, bottom], axis=1)

    padding_dimensions = (timesteps,) + extended_chorale.shape[1:] 

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    padding_start[:, :p_t] = 1
    padding_end[:, -p_b:] = 1

    extended_chorale = np.concatenate((padding_start,
                                       extended_chorale,
                                       padding_end),
                                       axis=0)

    if (dimension):
        return extended_chorale, p_t, p_b
    else:
        return extended_chorale


def label_conversion(labels, pad_to_num, timesteps, feature_num=352, base=88, mpe=False, onsets=False):
    '''
    This function is designed for MusicNet.
    Labels formatted with information of different instruments.
    
    See MusicNet/code/ProcessFeature.py for details about the format of labels stored in pickle files.
    '''
    
    assert(feature_num % base == 0)
    scale = feature_num // base
    spec_inst = MusicNet_Instruments
    inst_num  = get_instruments_num(spec_inst)
    mapping   = get_MusicNet_label_num_mapping(1, spec_inst)
    new_l     = np.zeros((len(labels), feature_num, inst_num+1))

    for t, label in enumerate(labels):
        if len(label.items()) == 0:
            new_l[t, :, 0] = 1
            continue

        for pitch, insts in label.items():
            for it in insts:
                if it not in mapping:
                    continue
                    
                ii = mapping[it]
                pr = range(pitch*scale, (pitch+1)*scale)
                if onsets:
                    new_l[t, pr, ii] = insts[it]
                else:
                    new_l[t, pr, ii] = 1

    if (((pad_to_num - new_l.shape[1]) % 2) == 0):
        p_t = (pad_to_num - new_l.shape[1]) // 2
        p_b = p_t

    else:
        p_t = (pad_to_num - new_l.shape[1]) // 2
        p_b = p_t + 1
    
    ext_dim = (new_l.shape[2],) if len(new_l.shape)==3 else None
    top    = np.zeros((new_l.shape[0], p_t)+ext_dim if ext_dim is not None else (new_l.shape[0], p_t))
    bottom = np.zeros((new_l.shape[0], p_b)+ext_dim if ext_dim is not None else (new_l.shape[0], p_b))
    new_l  = np.concatenate([top, new_l, bottom], axis=1)
    
    # This is for single channel output, by merging all the labels into one channel
    if mpe:
        mpe_l = np.nanmax(new_l[:,:,1:], axis=2)
        mpe_l = np.dstack((new_l[:,:,0], mpe_l)) 
        new_l = mpe_l
        
    return new_l

def load_hdf(data_path, inplace=False):
    if type(data_path) != list:
        data_path = [data_path]

    data = []
    for d in data_path:
        #print("Loading hdf file", d)
        dd = h5py.File(d, 'r')
        for i in range(len(dd)):
            data.append(dd[str(i)])
    if inplace:
        # Load data into RAM
        data = [dd[:] for dd in data]
            
    return data

def augment_hdf(data, chorale_index, time_index, timesteps, channels):

    feature = data[chorale_index]
    assert(feature.shape[2] >= len(channels)), "Channel of input feature must larger than the given channel length!\
            Length of feature channels: {}. Length of given channels: {}".format(feature.shape[2], len(channels))
    insert_range = range(16, 16+352)
    
    # For MusicNet, there are total 12 channels (harmonics/sub-harmonics of spec/ceps)
    # For MAPS, there are total 4 channels (Z, Spec, GCoS, Ceps)
    feature_48 = np.zeros((timesteps, 384, len(channels))) 
                
    if time_index < timesteps:
        overlap = timesteps-time_index
        feature_48[overlap:, insert_range] = feature[0:time_index][:,:,channels]

    elif time_index >= len(feature):
        overlap = len(feature)+timesteps-time_index-1
        feature_48[:overlap, insert_range] = feature[time_index-timesteps+1:][:,:,channels]
    else:
        start = time_index-timesteps+1
        feature_48[:, insert_range] = feature[start : start+timesteps][:,:,channels]
    
    return feature_48

def load_data(path):

    Y = []
    for d in path:
        print("Loading pickle", d)
        part = pickle.load(open(d, 'rb'))

        if type(part) == dict:
            # label loading
            tmp = []
            for key, value in part.items():
                tmp.append(value)
            part = tmp

        Y = np.concatenate([Y, part]) # Dim: pieces x frames x roll
    return Y
                 
            
def matrix_parser(m):
    x = np.zeros(shape=(m.shape[0], 2))

    for i in range(len(m)):
        if (np.sum(m[i]) != 0):
            x[i][0] = 1
            x[i][1] = midi2freq(np.argmax(m[i]) / 4 + 21)

    x[:, 1] = melody.hz2cents(x[:, 1])

    return x


def load_model(model_path):
    """


    """
    full_path = os.path.join(model_path, "arch.yaml")
    model = model_from_yaml(open(full_path).read())
    full_path = os.path.join(model_path, "weights.h5")
    model.load_weights(full_path)

    print("model " + model_path + " loaded")
    return model


def save_model(model, model_path, feature_type="CFP", input_channels=[1, 3], output_classes=2):
    # SAVE MODEL
    string = model.to_yaml()
    full_path = os.path.join(model_path, "arch.yaml")
    open(full_path, 'w').write(string)
    
    full_path = os.path.join(model_path, "configuration.csv")
    with open(full_path, "w", newline='') as config:
        writer = csv.writer(config)
        writer.writerow(["Model name", "Feature type", "Input channels", "Output classes"])
        
        model_name = model_path[model_path.rfind("/")+1:]
        writer.writerow([model_name, feature_type, input_channels, output_classes])
    
    print("model " + model_name + " saved")

def model_info(model_path):
    config_file = os.path.join(model_path, "configuration.csv")
    with open(config_file, "r", newline='') as config:
        reader = csv.DictReader(config)
        row = next(iter(reader))
        f_type, channels, out_classes = row["Feature type"], row["Input channels"], row["Output classes"]
        
        channels = ast.literal_eval(channels)
        out_classes = int(out_classes)
        
    return f_type, channels, out_classes

def model_copy(origin, target):

    for index, layer in enumerate(target.layers):
        if layer.__class__.__name__ == 'LSTM':
            weights = origin.layers[index].get_weights()
            units = weights[1].shape[0]
            bias = weights[2]
            if len(bias) == units * 8:
                # reshape the kernels
                kernels = np.split(weights[0], 4, axis=1)
                kernels = [kernel.reshape(-1).reshape(kernel.shape, order='F') for kernel in kernels]
                weights[0] = np.concatenate(kernels, axis=1)

                # transpose the recurrent kernels
                recurrent_kernels = np.split(weights[1], 4, axis=1)
                recurrent_kernels = [kernel.T for kernel in recurrent_kernels]
                weights[1] = np.concatenate(recurrent_kernels, axis=1)

                # split the bias into half and merge
                weights[2] = bias[:units * 4] + bias[units * 4:]
                layer.set_weights(weights)
                print("Set success")
        else:
            layer.set_weights(origin.layers[index].get_weights())
