
import os
import csv
import h5py
import argparse
import numpy as np
import soundfile as sf

from MusicNet.ProcessLabels import manage_label_process
from MusicNet.MelodyExt import feature_extraction


def fetch_harmonic(data, cenf, ith_har, 
                   start_freq=27.5, 
                   num_per_octave=48, 
                   is_reverse=False):
    
    ith_har += 1
    if ith_har != 0 and is_reverse:
        ith_har = 1/ith_har
    
    #harmonic_series = [12, 19, 24, 28, 31]     
    bins_per_note = int(num_per_octave / 12)           
    total_bins = int(bins_per_note * 88)
    
    hid = min(range(len(cenf)), key=lambda i: abs(cenf[i]-ith_har*start_freq))
    
    harmonic = np.zeros((total_bins, data.shape[1]))
    upper_bound = min(len(cenf)-1, hid+total_bins)
    harmonic[:(upper_bound-hid)] = data[hid:upper_bound]
    
    return harmonic
    
                    

  
def make_dataset_audio(dataset_name, 
                       song_list, 
                       fmt=".hdf",
                       harmonic=False,
                       num_harmonic=0):
    
    fs = 44100
    if harmonic:
        freq_range = [1.0, fs/2]
    else:
        freq_range = [27.5, 4487.0]
    
    hdf_out = h5py.File(dataset_name+fmt, "w")

    
    
    for idx, song in enumerate(song_list):
        print("Extracting({}/{}): {}".format(idx+1, len(song_list), song))
        
        out  = feature_extraction(song, fc=freq_range[0], tc=(1/freq_range[1]), Down_fs=fs)
        cenf = out[5]
        #z, spec, gcos, ceps, cenf = out[0:5]

        piece = np.transpose(np.array(out[0:4]), axes=(2, 1, 0))
        
        if harmonic:
            har = []
            for i in range(num_harmonic+1):
                har.append(fetch_harmonic(out[1], cenf, i))
            har_s = np.transpose(np.array(har), axes=(2, 1, 0))
            
            har = []
            for i in range(num_harmonic+1):
                har.append(fetch_harmonic(out[3], cenf, i, is_reverse=True))
            har_c = np.transpose(np.array(har), axes=(2, 1, 0))
            
            piece = np.dstack((har_s, har_c))
        
        hdf_out.create_dataset(str(idx), data=piece, compression="gzip", compression_opts=5)
    
    hdf_out.close()    
    
    #print("Saving files...")
    #base = dataset_name.rfind("/")
    #f_name = dataset_name[base+1:] + fmt
    #base = dataset_name[:base]
    #pickle.dump(Z, open(os.path.join(base, "z", f_name), 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(Spec, open(os.path.join(base, "spec", f_name), 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(Gcos, open(os.path.join(base, "gcos", f_name), 'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(Ceps, open(os.path.join(base, "ceps", f_name), 'wb'), pickle.HIGHEST_PROTOCOL)
        

def Manage_Feature_Process(audio_path, 
                           save_path,
                           save_name,
                           fmt          = ".hdf",
                           num_per_file = 40,
                           harmonic     = True,
                           permutate    = False):        
    
     
    song_list = os.listdir(audio_path)
    song_list = [os.path.join(audio_path, ss) for ss in song_list]
    if permutate:
        song_list = np.random.permutation(song_list)
    
    with open(os.path.join(save_path, "SongList.csv"), "w", newline='') as csvList:
        writer = csv.writer(csvList)
        writer.writerow(["File name", "id", "Save path"])
        
        files_num = int(len(song_list)/num_per_file)
        files_num = 1 if files_num == 0 else files_num
        for i in range(files_num):
            print("Files {}/{}".format(i+1, files_num))
            
            sub_list = song_list[(i*num_per_file):((i+1)*num_per_file)]
            sub_name = os.path.join(save_path, (save_name+"_"+str(i+1)))
            make_dataset_audio(sub_name, sub_list, fmt, harmonic=harmonic, num_harmonic=5)    
            
            
            # write record to file
            for id in sub_list:
                f_name = save_name+"_"+str(i+1)
                id = id[(id.rfind("/")+1):]
                writer.writerow([f_name, id, save_path])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Program to process MusicNet features for training and testing.")
    parser.add_argument("--MusicNet-path", 
                        help="Path to your downloaded MusicNet folder path. (default: %(default)s)",
                        type=str, default="./")
    parser.add_argument("--train-num-per-file", 
                        help="Number of pieces to be stored in each output train file (default: %(default)d)",
                        type=int, default=20)
    parser.add_argument("--test-num-per-file", 
                        help="Number of pieces to be stored in each output test file (default: %(default)d)",
                        type=int, default=10)
    parser.add_argument("--save-path",
                        help="Path for saving the output files (default: <MusicNet_path>/features)",
                        type=str)
    parser.add_argument("--no-harmonic",
                        help="Wether to generate harmonic features (HCFP, 12 channels) or CFP features only(4 channels).",
                        action="store_true")
    args = parser.parse_args()
    
    
    MusicNet_path = args.MusicNet_path
    
    train_audio_path = os.path.join(MusicNet_path, "train_data")
    test_audio_path  = os.path.join(MusicNet_path, "test_data")
    train_label_path = os.path.join(MusicNet_path, "train_labels")
    test_label_path  = os.path.join(MusicNet_path, "test_labels")
    
    if args.save_path is None:
        args.save_path = os.path.join(MusicNet_path, "./features")

    train_save_path = os.path.join(args.save_path, "train")
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    test_save_path = os.path.join(args.save_path, "test")
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    
    train_save_name = "train_" + str(args.train_num_per_file)
    test_save_name  = "test_" + str(args.test_num_per_file)
    
    # Process training data and label
    #Manage_Feature_Process(train_audio_path, train_save_path, train_save_name,
    #                       num_per_file=args.train_num_per_file, harmonic=not args.no_harmonic)
    manage_label_process(train_label_path, args.train_num_per_file, train_save_path, t_unit=0.02)
    
    # Process testing data and label
    #Manage_Feature_Process(test_audio_path, test_save_path, test_save_name, 
    #                       num_per_file=args.test_num_per_file, harmonic=not args.no_harmonic)
    #manage_label_process(test_label_path, args.test_num_per_file, test_save_path, t_unit=0.02)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
