

import os
import csv
import h5py
import pickle
import argparse
import numpy as np

from tqdm import trange

from project.test import inference
from project.utils import label_conversion, load_model, model_info





def roll_down_sample(data, threshold=0.5, occur_num=2, base=88):
    total_roll = data.shape[1]
    assert total_roll % base == 0, "Wrong length: {}, {} % {} should be zero!".format(total_roll, total_roll, base)
    
    scale = round(total_roll/base)
    assert(occur_num>0 and occur_num<scale)
    
    
    return_v = np.zeros((len(data), base), dtype=int)
    
    for i in range(0, data.shape[1], scale):
        total = np.sum(data[:, i : i+scale], axis=1)
        return_v[:, int(i/scale)] = np.where(total>threshold*occur_num, total/occur_num, 0)
    return_v = np.where(return_v>1, 1, return_v)
        
    return return_v



def predict(feature, 
            model, 
            threshold = None,
            full_predict=True, 
            MAX_FRAME=9000,
            channels=[0],
            instruments=1):
    
    original_v = True if threshold==None else False
    #if original_v:
    #    print("Keeping original prediction value")
    
    if len(feature) > MAX_FRAME:
        overlap = 4
        
        len_f = len(feature)
        turns = int(np.ceil(len_f/MAX_FRAME))
        pred  = []
        #print("Total sub rounds: " + str(turns))
        
        padding = np.zeros(((overlap,)+feature.shape[1:]))
        feature = np.concatenate((padding, feature, padding), axis=0)

        for j in trange(turns, desc='A piece', leave=False):
            if j != (turns-1):
                sub_feature = feature[j*MAX_FRAME : (j+1)*MAX_FRAME+2*overlap]
            else:
                sub_feature = feature[j*MAX_FRAME : ]

            tmp_pred = inference(feature = sub_feature[:, :, channels],
                                 model = model,
                                 threshold = threshold,
                                 isMPE = True,
                                 original_v=original_v,
                                 channel=len(channels),
                                 instruments=instruments,
                                 keep_progress=False)
            if j == 0:
                pred = tmp_pred[overlap:-overlap]
            else:
                pred = np.concatenate((pred, tmp_pred[overlap:-overlap]), axis=0)
            
            if not full_predict:
                break;

    else:
        pred = inference(feature = feature[:, :, channels],
                         model = model,
                         threshold = threshold,
                         isMPE = True,
                         original_v=original_v,
                         channel=len(channels),
                         instruments=instruments)
    return pred

def parse_path(path, label=False):
    data = []
    if os.path.isdir(path):
        files = os.listdir(path)
        for ff in files:
            if label and "label" in ff:
                data.append(ff)
            elif "label" not in ff:
                data.append(ff)
        data = [os.path.join(path, dd) for dd in data]
    elif os.path.isfile(path):
        data.append(path)
    else:
        assert(False), "The given path is neither a directory nor a file, please check again. \
                        Path: " + path
    return data
 
def load_files(paths, use_ram=True):
    data = []
    for path in paths:
        # Get extension
        ext = path[path.rfind(".")+1:]
        if ext=="hdf":
            ff = h5py.File(path, "r")
            for key in ff:
                data.append(ff[key])
            
            if use_ram:
                data = [dd[:] for dd in data]
            
        # If the extension is pickle, then it also load data into the ram.
        elif ext=="pickle":
            tmp_dd = pickle.load(open(path, 'rb'))
            for dd in tmp_dd:
                data.append(dd)
        else:
            assert("False"), "No matching files with extension '{}'".format(ext)

    return data
 

 
def FullTest(model_path, test_path, 
             label_path = None,
             pred_save_path="./predictions",
             use_ram = True,
             MAX_FRAME=1800):
    
    
    
    # Load files
    print("Loading files")
    features = parse_path(test_path)
    if label_path is not None:
        # Assume there are exactly label files corresponding to the test audios
        #labels = parse_path(label_path, label=True)
        labels = []
        for ff in features:
            ext = ff[ff.rfind("."):]
            if ext != ".hdf":
                continue
            if "_label" not in ff:
                ll = ff[:ff.rfind(".")] + "_label.pickle"
                labels.append(ll)
        labels = load_files(labels, use_ram=use_ram)

    features = load_files(features, use_ram=use_ram)
    model    = load_model(model_path)    

    
    # Validate on model/feature configurations
    f_type, channels, out_classes = model_info(model_path)
    
    if f_type=="HCFP" and features[0].shape[2] < 12:
        assert(False), "The model uses HCFP as input feature, but loaded features are not."
    if f_type=="CFP" and features[0].shape[2] == 12:
        assert(len(channels)==2 and 1 in channels and 3 in channels), """The 
             The given feature are HCFP, but the model uses more feature types.
             Model input feature types: """ + str(channels) + " ({0: Z, 1: Spec, 2: GCoS, 3: Ceps})"
        channels = [0, 6]
    mpe = False
    if out_classes==2:
        mpe = True
    
    
    
    # To avoid running out of memory. 
    # 9000 is suitable for 32G RAM with one instrument only and all 4 channels used. (Max ram usage almost 100%)
    #MAX_FRAME = 1800 
    print("Max frame per prediction: ", MAX_FRAME)

    
    
    # Start to predict
    pred_out = h5py.File(os.path.join(pred_save_path, "pred.hdf"), "w")
    label_out = h5py.File(os.path.join(pred_save_path, "label.hdf"), "w")
    len_data = len(features)
    for i in trange(len_data, desc='Dataset'):
        feature = features[0][:]
        
        pred = predict(feature, model, MAX_FRAME=MAX_FRAME, channels=list(channels), instruments=out_classes-1)
        
        # Save to output
        pred_out.create_dataset(str(i), data=pred, compression="gzip", compression_opts=5)
        del feature, features[0]
        
        # Process corresponding label
        if label_path is not None:
            ll = labels[0]
            if type(ll) != np.ndarray:
                ll = label_conversion(ll, 352, 128, mpe=mpe)[:, :, 1:]
            label_out.create_dataset(str(i), data=ll, compression="gzip", compression_opts=5)
            del labels[0]
        
        
        
    pred_out.close()
    label_out.close()
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test on the audio in the given path.")
    parser.add_argument("--model-path", 
                        help="Path to the pre-trained model.",
                        type=str)
    parser.add_argument("--test-path",
                        help="Path to the pre-precessed features of test set. Could be a directory or a file",
                        type=str)
    parser.add_argument("--label-path",
                        help="Path to the pre-precessed test label features. If not given, the output won't include \
                        labels and it will be hard to use the predictions in the future.",
                        type=str)
    parser.add_argument("--pred-save-path",
                        help="Path for the output predictions to save (default %(default)s/<Model_Name>)",
                        type=str, default="./predictions")
    parser.add_argument("--use-ram",
                        help="Wether to load all the data into ram. (default %(default)s)",
                        type=bool, default=True)
    args = parser.parse_args()
    

    model_name = args.model_path[args.model_path.rfind("/")+1:]
    args.pred_save_path = os.path.join(args.pred_save_path, model_name)
    if not os.path.exists(args.pred_save_path):
        os.makedirs(args.pred_save_path)

    FullTest(model_path     = args.model_path, 
             test_path      = args.test_path,
             label_path     = args.label_path,
             pred_save_path = args.pred_save_path,
             use_ram        = args.use_ram,
             MAX_FRAME      = 1800)
             
    print("Finished")
 
             
    
   
    
    
    

