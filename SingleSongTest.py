
import os
import argparse
import numpy as np

from PrintPianoRoll import PLOT
from Predict import model_info, predict
from MusicNet.FeatureExtraction import fetch_harmonic

from project.utils import load_model
from project.MelodyExt import feature_extraction
from project.configuration import MusicNet_Instruments


"""
def get_file_by_size(directory, num=0):
    
    # Get all files.
    list = os.listdir(directory)

    # Loop and add files to list.
    pairs = []
    for file in list:

        # Use join to get full file path.
        location = os.path.join(directory, file)

        # Get size and add to list of tuples.
        size = os.path.getsize(location)
        pairs.append((size, location))

    # Sort list of tuples by the f
    pairs.sort(key=lambda s: s[0])
    
    return pairs[num][1]
    
def single_song_test():
    test_path = "/media/whitebreeze/本機磁碟/maps/DATASET_Train/wav"
    
    test_audio = get_file_by_size(test_path, 2)

    feature = feature_extraction(test_audio)
    feature = np.transpose(feature[0:4], axes=(2, 1, 0))
    
    

    model = load_model("./onsets_model")

    print(feature[:, :, 0].shape)
    extract_result = inference(feature= feature[:, :, 0],
                               model = model,
                               batch_size=10, 
                               isMPE = True,
                               original_v = True).transpose()

    
    print("Average: {}".format(np.mean(extract_result)))
    result = []
    result.append(np.where(extract_result>0.3, 1, 0))
    result.append(np.where(extract_result>0.4, 1, 0))
    #result.append(np.where(extract_result>MAX_V*0.5, 1, 0))
    result.append(roll_down_sample(result[-1].transpose()).transpose())
    
    fig, ax = plt.subplots(nrows=len(result))
    ax[0].imshow(result[0], aspect='auto', origin='lower', cmap="PuBuGn")
    ax[1].imshow(result[1], aspect='auto', origin='lower', cmap="OrRd")
    ax[2].imshow(result[2], aspect='auto', origin='lower', cmap="RdPu")
    plt.show()
    plt.savefig('./result.png', box_inches='tight', dpi=250)
    
    return centFreq
    
"""


def main(args):
    # Pre-process features
    assert(os.path.isfile(args.input_audio)), "The given path is not a file!. Please check your input again."
    Z, tfrL0, tfrLF, tfrLQ, t, cenf, f = feature_extraction(args.input_audio)
    
    # Post-process feature according to the configuration of model
    feature_type, channels, out_class = model_info(args.model_path)
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
        
        feature = np.array([Z, tfrL0, tfrLF, tfrLQ])[channels]
        feature = np.transpose(feature, axes=(2, 1, 0))
        
    model = load_model(args.model_path)
    
    
    pred = predict(feature, model,
                   channels=channels,
                   instruments=out_class-1)
    
    
    # Print figure
    base_path = args.input_audio[:args.input_audio.rfind("/")]
    save_name = os.path.join(base_path, args.output_fig_name)
    
    plot_range = range(500, 1500)
    if max(plot_range) > len(pred):
        plot_range = range(0, len(pred))
    pred = pred[plot_range]
    
    if out_class >= 11:
        assert(out_class==12), "There is something wrong with the configuration. \
                                Expected value: 12, Current value: {}".format(out_class)
        titles = MusicNet_Instruments
    else:
        assert(out_class==2), "There is something wrong with the configuration. \
                               Expected value: 2, Current value: {}".format(out_class)
        titles = ["Piano"]
    
    PLOT(pred, save_name, plot_range, titles=titles)


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
    args = parser.parse_args()
    args.num_harmonics = 5
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    