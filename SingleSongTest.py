import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import argparse
import numpy as np
import h5py

from project.Feature.FeatureFirstLayer import feature_extraction
from project.Feature.FeatureSecondLayer import fetch_harmonic
from project.Predict import predict, predict_v1
from project.postprocess import MultiPostProcess

from project.utils import load_model, model_info
from project.configuration import MusicNet_Instruments


def main(args):
    # Pre-process features
    assert(os.path.isfile(args.input_audio)), "The given path is not a file!. Please check your input again. Given input: {}".format(audio.input_audio)
    print("Processing features of input audio: {}".format(args.input_audio))
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
    #pred = predict(feature[:,:,channels], model, timesteps, out_class, batch_size=4, overlap_ratio=2/4)
    pred = predict_v1(feature[:,:,channels], model, timesteps, batch_size=4)
    
    #p_out = h5py.File("pred.hdf", "w")
    #p_out.create_dataset("0", data=pred)
    #p_out.close()

    midi = MultiPostProcess(pred, mode="note", onset_th=args.onset_th, dura_th=0.5, frm_th=3, inst_th=1.1, t_unit=0.02)
    
    if args.to_midi is not None:
        midi.write(args.to_midi)
        print("Midi written as {}".format(args.to_midi))

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
    parser.add_argument("--onset-th", help="Onset threshold (5~8)",
                        type=float, default=7)
    args = parser.parse_args()
    args.num_harmonics = 5
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
