import os
import ast
import csv
import h5py
import numpy as np

from scipy.special import expit
from project.configuration import get_MusicNet_label_num_mapping, get_instruments_num, MusicNet_Instruments
from keras.models import model_from_json, model_from_yaml

from keras import layers as L
from keras.utils import multi_gpu_model
from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d
from project.Models.model_attn import seg, multihead_attention

TIMESTEP = 128

def sigmoid(x):
    #return 1 / (1 + np.exp(-x))
    return expit(x)

def label_conversion(label, tid, 
                     timesteps=128, 
                     ori_feature_size=352, 
                     feature_num=384, 
                     base=88, 
                     mpe=False, 
                     onsets=False,
                     offsets=False):
    '''
    This function is designed for MusicNet.
    Labels formatted with information of different instruments.
    
    See MusicNet/code/ProcessFeature.py for details about the format of labels stored in pickle files.
    '''

    labels = label[tid:(tid+timesteps)]
    if len(labels) < timesteps:
        for _ in range(timesteps-len(labels)):
            labels.append({})
    
    assert(ori_feature_size % base == 0)
    scale = feature_num // base
    spec_inst = MusicNet_Instruments
    inst_num  = get_instruments_num(spec_inst)
    mapping   = get_MusicNet_label_num_mapping(1, spec_inst)
    new_l     = np.zeros((len(labels), ori_feature_size, inst_num+1))
    new_l[:,:,0] = 1

    for t, label in enumerate(labels):
        if len(label.items()) == 0:
            continue

        for pitch, insts in label.items():
            for it in insts:
                if it not in mapping:
                    continue
                    
                ii = mapping[it]
                pr = range(pitch*scale, (pitch+1)*scale)
                #new_l[t, pr, 0] = 0
                if onsets:
                    new_l[t, pr, ii] = insts[it][0]
                elif offsets:
                    new_l[t, pr, ii] = insts[it][1]
                else:
                    new_l[t, pr, ii] = 1
    new_l[:,:,0] = 1 - new_l[:,:,1]

    h = new_l.shape[1]
    p_b = (feature_num-h) // 2
    p_t = feature_num - p_b - new_l.shape[1]
    
    b_shape = (new_l.shape[0], p_b)
    t_shape = (new_l.shape[0], p_t)
    if len(new_l.shape) == 3:
        b_shape += (new_l.shape[2],)
        t_shape += (new_l.shape[2],)
    bottom = np.zeros(b_shape)
    top = np.zeros(t_shape)
    new_l  = np.concatenate([bottom, new_l, top], axis=1)
    
    # This is for single channel output, by merging all the labels into one channel
    if mpe:
        mpe_l = np.nanmax(new_l[:,:,1:], axis=2)
        new_l = np.dstack((new_l[:,:,0], mpe_l))
        
    return new_l
                         
def load_model(model_path):
    custom_layers = {
        "multihead_attention": multihead_attention,
        "Conv2D": L.Conv2D,
        "split_heads_2d": split_heads_2d,
        "local_attention_2d": local_attention_2d,
        "combine_heads_2d": combine_heads_2d
    }
    model = model_from_yaml(open(os.path.join(model_path, "arch.yaml")).read(), custom_objects=custom_layers)

    full_path = os.path.join(model_path, "weights.h5")
    with h5py.File(full_path, "r") as w:
        keys = list(w.keys())
        is_para = any(["model" in k for k in keys])

    if is_para:
        para_model = multi_gpu_model(model, gpus=2)
        para_model.load_weights(full_path)
        print("Model {} loaded. Using multi-GPU to train".format(model_path))

        return para_model
    
    model.load_weights(full_path)
    print("model " + model_path + " loaded")

    return model

def save_model(model, model_path, feature_type="CFP", channels=[1, 3], output_classes=2, timesteps=128):
    # SAVE MODEL
    string = model.to_yaml()
    full_path = os.path.join(model_path, "arch.yaml")
    open(full_path, 'w').write(string)
    
    full_path = os.path.join(model_path, "configuration.csv")
    with open(full_path, "w", newline='') as config:
        writer = csv.writer(config)
        writer.writerow(["Model name", "Feature type", "Input channels", "Output classes", "Timesteps"])
        
        model_name = model_path[model_path.rfind("/")+1:]
        writer.writerow([model_name, feature_type, channels, output_classes, timesteps])
    
    print("model " + model_name + " saved")

def model_info(model_path):
    config_file = os.path.join(model_path, "configuration.csv")
    with open(config_file, "r", newline='') as config:
        reader = csv.DictReader(config)
        row = next(iter(reader))
        f_type, channels, out_classes, timesteps = row["Feature type"], row["Input channels"], row["Output classes"], row["Timesteps"]
        
        channels = ast.literal_eval(channels)
        out_classes = int(out_classes)
        timesteps = int(timesteps)
        
    return f_type, channels, out_classes, timesteps

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
