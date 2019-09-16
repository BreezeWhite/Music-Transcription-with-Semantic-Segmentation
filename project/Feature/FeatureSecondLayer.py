import h5py
import numpy as np

from project.Feature.FeatureFirstLayer import feature_extraction


def process_feature_song_list(
        dataset_name, 
        song_list,
        harmonic=False,
        num_harmonic=0
    ):

    fs = 44100
    if harmonic:
        freq_range = [1.0, fs/2]
    else:
        freq_range = [27.5, 4487.0]
    hdf_out = h5py.File(dataset_name+".hdf", "w")
    
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

def fetch_harmonic(
        data, 
        cenf, 
        ith_har, 
        start_freq=27.5, 
        num_per_octave=48, 
        is_reverse=False
    ):
    
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

