
import os
import pickle
import h5py
import librosa
import numpy as np
import soundfile as sf

from project.MelodyExt import feature_extraction



def LoadFile(fileName, loadLabel = True):
    if(not os.path.isfile(fileName)):
        print("File not found! {}".format(fileName))
    
    maps = h5py.File(fileName,'r')
    data = maps['imdb']['images']['data'][:].squeeze()
    label = None
    if loadLabel:
        label = maps['imdb']['images']['labels'][:]
        if label.shape[1] > 88:
            label = label[:,21:109]
        label = np.where(label.squeeze()>0, 1, 0)
    maps.close()
    
    data = data.reshape((data.shape[0], 1, 1, data.shape[1]))
    
#    onsets_channel = onsets_feature(data)
#    data = np.concatenate((data, onsets_channel), axis=2)
    
    return data, label


def label_parse(song, len_t, ori_label_path):
    # Name conversion
    audio = song.split("/")[-1]
    front = audio.split("_")[-1]
    front = front.split(".")[0]
    front = front + "_MUS_"
    name = audio.split(".wav")[0]
    name = front + name + ".mat"
    final = os.path.join(ori_label_path, name)

    data, label = LoadFile(final)

    # Process label
    label_mat = np.zeros((len_t, 352))
    for i in range(len_t):
        if (i*2  > len(label)) or (i*2+1 > len(label)): 
            break

        roll1 = label[i*2]
        roll2 = label[i*2+1]
        rollf = roll1 | roll2
        
        for j, val in enumerate(rollf):
            if val == 0:
                continue
            #label_mat[correspond_id[j]] = 1
            bb = j*4
            rr = range(bb, bb+4)
            label_mat[i, rr] = 1

    return label_mat
  
def make_dataset_audio(dataset_name, 
                       song_list, 
                       fmt=".pickle",
                       ori_label_path="/media/whitebreeze/本機磁碟/maps/Full Extraction/GCoS"):
    
    X = []
    Y = []
    for i, song in enumerate(song_list):
        print("Extracting({}/{}): {}".format(i+1, len(song_list), song))
        
        out   = feature_extraction(song)
        score = np.transpose(out[0:4], axes=(2, 1, 0))
        X.append(score)
        
        score = label_parse(song, len(out[4]), ori_label_path)
        Y.append(score)
        
    pickle.dump(X, open(dataset_name+fmt, 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(Y, open(dataset_name+"_label"+fmt, "wb"), pickle.HIGHEST_PROTOCOL)
    print(str(len(X)) + ' files written in ' + dataset_name)    

def Manage_Feature_Process(audio_path, 
                           save_path,
                           save_name,
                           fmt=".pickle",
                           num_per_file=30):        
    
    
    song_list = os.listdir(audio_path)
    song_list = [os.path.join(audio_path, ss) for ss in song_list]
    
    
    
    
    files_num = int(len(song_list)/num_per_file)
    for i in range(2, files_num):
        print("Files {}/{}".format(i+1, files_num))
        
        sub_list = song_list[(i*num_per_file):((i+1)*num_per_file)]
        sub_name = os.path.join(save_path, (save_name+"_"+str(i+1)))
        make_dataset_audio(sub_name, sub_list, fmt)    





if __name__ == "__main__":
    train_audio_path = "/media/whitebreeze/本機磁碟/maps/DATASET_Train/wav"
    test_audio_path  = "/media/whitebreeze/本機磁碟/maps/DATASET_Eval/wav"
    
    save_path       = "/media/whitebreeze/本機磁碟/maps"
    train_save_name = "train_30"
    test_save_name  = "test_10"
    
    #Manage_Feature_Process(train_audio_path, save_path, train_save_name)
    Manage_Feature_Process(test_audio_path, save_path, test_save_name, num_per_file=10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
