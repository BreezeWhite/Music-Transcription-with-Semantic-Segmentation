import sys
sys.path.append("../MusicNet")

import glob, os
import pickle
from FeatureExtraction import Manage_Feature_Process
from MAPS_ProcessLabel import ProcessLabel



def list_wavs(path):
    if type(path) == str:
        return glob.glob(os.path.join(path, "*.wav"))
    
    assert(type(path)==list)
    files = []
    for p in path:
        ff = glob.glob(os.path.join(p, "*.wav"))
        files.append(ff)
        
    return files

def Manage_Process_Label(files, save_path, num_per_file, t_unit=0.02):
    
    files = [ff.replace(".wav", ".txt") for ff in files]
    
    iters = np.ceil(len(files)/num_per_file).astype('int')
    for i in trange(iters):
        sub_files = files[i*num_per_file : (i+1)*num_per_file]
        
        labels = []
        for sf in sub_files:
            labels.append(ProcessLabel(sf, t_unit=t_unit))

        f_name = "train" if "train" in files[0] else "test"
        post   = "_{}_{}_label.pickle".format(num_per_file, i+1)
        f_name += post
        f_name = os.path.join(save_path, f_name)

        pickle.dump(labels, open(f_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    
def main(args):
    train_folders = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "SptkBGAm", "SptkBGCl"]
    test_folders  = ["ENSTDkAm", "ENSTDkCl"]
    train_folders = [os.path.join(args.MAPS_path, ff, "MUS") for ff in train_folders]
    test_folders  = [os.path.join(args.MAPS_path, ff, "MUS") for ff in test_folders]
    
    train_audios = list_wavs(train_folders)
    test_audios  = list_wavs(test_folders)
    
    train_save_path = os.path.join(args.save_path, "train")
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    test_save_path = os.path.join(args.save_path, "test")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
        
    train_save_name = "train_" + str(args.train_num_per_file)
    test_save_name  = "test_" + str(args.test_num_per_file)
    
    # Process training features
    Manage_Feature_Process(train_audios, train_save_path, train_save_name, num_per_file=args.train_num_per_file)
    Manage_Process_Label(train_audios, args.train_save_path, args.train_num_per_file)
    
    # Process testing features
    Manage_Feature_Process(test_audios, test_save_path, test_save_name, num_per_file=args.test_num_per_file)
    Manage_Process_Label(test_audios, args.test_save_path, args.test_num_per_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Program to process MAPS features for training and testing.")
    parser.add_argument("--MAPS-path", 
                        help="Path to your downloaded MAPS folder path. (default: %(default)s)",
                        type=str, default="./")
    parser.add_argument("--train-num-per-file", 
                        help="Number of pieces to be stored in each output train file (default: %(default)d)",
                        type=int, default=20)
    parser.add_argument("--test-num-per-file", 
                        help="Number of pieces to be stored in each output test file (default: %(default)d)",
                        type=int, default=10)
    parser.add_argument("--save-path",
                        help="Path for saving the output files (default: <MAPS>/features)",
                        type=str)
    parser.add_argument("--no-harmonic",
                        help="Wether to generate harmonic features (HCFP, 12 channels) or CFP features only(4 channels).",
                        action="store_true")
    args = parser.parse_args()
    
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
