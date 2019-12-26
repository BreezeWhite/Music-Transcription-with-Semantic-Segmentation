
import sys
sys.path.append("MusicNet/")

import os
import argparse
import numpy as np
import h5py

from PrintPianoRoll import PLOT
from Predict import predict
from MusicNet.FeatureExtraction import fetch_harmonic
#from Evaluation import peak_picking

from project.utils import load_model, to_midi, model_info
from project.MelodyExt import feature_extraction
from project.configuration import MusicNet_Instruments



def main(args):
    # Pre-process features
    assert(os.path.isfile(args.input_audio)), "The given path is not a file!. Please check your input again."
    print("Processing features")
    Z, tfrL0, tfrLF, tfrLQ, t, cenf, f = feature_extraction(args.input_audio)
    
    # Post-process feature according to the configuration of model
    feature_type, channels, out_class, timesteps = model_info(args.model_path)
    if feature_type == "HCFP":
        assert(len(channels) == (args.num_harmonics*2+2))
        
        spec = []
        ceps = []
        for i in range(args.num_harmonics):
            spec.append(fetch_harmonic(tfrL0, cenf, i))
            ceps.append(fetch_harmonic(tfrLQ, cenf, i))
        
        spec = np.transpose(np.array(spec), axes=(2, 1, 0))
        ceps = np.transpose(np.array(ceps), axes=(2, 1, 0))
        
        feature = np.dstack((spec, ceps))
    else:
        assert(len(channels) <= 4)
        
        feature = np.array([Z, tfrL0, tfrLF, tfrLQ])
        feature = np.transpose(feature, axes=(2, 1, 0))

    model = load_model(args.model_path)
    

    print("Predicting...")
    pred = predict(feature, model,
                   timesteps=timesteps,
                   channels=channels,
                   instruments=out_class-1)
    p_out = h5py.File("pred.hdf", "w")
    p_out.create_dataset("0", data=pred)
    p_out.close()

    for i in range(pred.shape[2]):
        pred[:,:88,i] = peak_picking(pred[:,:,i])
    pred = pred[:,:88]

    
    # Print figure
    base_path = args.input_audio[:args.input_audio.rfind("/")]
    save_name = os.path.join(base_path, args.output_fig_name)
    
    plot_range = range(500, 1500)
    if max(plot_range) > len(pred):
        plot_range = range(0, len(pred))
    pp = pred[plot_range]
    
    if out_class >= 11:
        assert(out_class==12), "There is something wrong with the configuration. \
                                Expected value: 12, Current value: {}".format(out_class)
        titles = MusicNet_Instruments
    else:
        assert(out_class==2), "There is something wrong with the configuration. \
                               Expected value: 2, Current value: {}".format(out_class)
        titles = ["Piano"]
    
    print("Ploting figure...")
    #PLOT(pp, save_name, plot_range, titles=titles)
    print("Output figure to {}".format(base_path))
    
    if args.to_midi is not None:
        midi_path = args.to_midi
        
        threshold = [0.45, 0.5]
        for th in threshold:
            midi = to_midi(pred, midi_path+"_"+str(th), threshold=th)

            roll = midi.get_piano_roll()
            print("Shape of output midi roll: ", roll.shape)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Transcribe on the given audio.")
    parser.add_argument("-i", "--input-audio",
                        help="Path to the input audio you want to transcribe",
                        type=str)
    parser.add_argument("-m", "--model-path", 
                        help="Path to the pre-trained model.",
                        type=str)
    parser.add_argument("-o", "--output-fig-name",
                        help="Name of transcribed figure of piano roll to save.",
                        type=str, default="Piano Roll")
    parser.add_argument("--to-midi", help="Also output the transcription result to midi file.",
                        type=str)
    args = parser.parse_args()
    args.num_harmonics = 5
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
