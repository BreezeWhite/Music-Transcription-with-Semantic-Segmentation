import pickle
import tqdm
import numpy as np
import argparse
import os
import yaml, csv
from project.MelodyExt import feature_extraction
from project.utils import freq2midi
from project.midi_handler import midi2score

def datasets_importer(data_DIR, data_type):
    sets = []
    print("Load from %s" % data_DIR)
    for filename in os.listdir(data_DIR):
          if(data_type in filename):
            sets.append(data_DIR+filename)
    print("Load successfully!")
    return sets


def medleydb_preprocessing(song_list,
                           path="/media/timlu/本機磁碟/MedleyDB/Audio/",
                           label_path="/media/timlu/本機磁碟/MedleyDB/Annotations/Pitch_Annotations",
                           rank_path="/media/timlu/本機磁碟/MedleyDB/Annotations/Stem_Rankings"
                           ):
    file_list = []
    vocal_track_list = []
    rank_list = []
    for dir_name in os.listdir(path):
        clean = True
        if ("." not in dir_name):
            for song in song_list:
                if (dir_name in song):
                    clean = False
                    break
            if (clean == False):
                with open(path + dir_name + "/" + dir_name + "_METADATA.yaml", 'r') as stream:
                    s = yaml.load(stream)
                    track_list = []
                    vocal = False
                    for k in s["stems"].keys():
                        if ("singer" in s["stems"][k]['instrument']):
                            vocal = True
                            track_list.append(k)
                    if (vocal == True):
                        file_list.append(path + dir_name + "/" + dir_name + "_MIX.wav")

                        vocal_track_list.append([label_path + "/" + dir_name, track_list])

                        r = np.loadtxt(open(rank_path + "/" + dir_name + "_RANKING.txt", 'rb'), delimiter=",",
                                       dtype="str")

                        rank_list.append(r)

    return file_list, vocal_track_list, rank_list


def label_parser(label, data, vocal_track_list=None):
    # parser label for mir1k
    if ("mir1k" in data):
        score = np.loadtxt(label)
        score_mat = np.zeros((len(score), 352))
        for i in range(score_mat.shape[0]):
            n = score[i]
            if (n != 0):
                score_mat[i, int(np.round((score[i] - 21) * 4))] = 1
    # parser label for medleydb
    elif ("medleydb" in data):
        # TODO: vocal ranking handle (only one exception in medleydb dataset)
        score = np.loadtxt(open(label, 'rb'), delimiter=",")
        score_mat_tmp = np.zeros((int(score[-1][0] // 0.0058) + 20, 352))

        for t in vocal_track_list[1]:
            try:
                with open(vocal_track_list[0] + "_STEM_" + t[1:] + ".csv") as csvfile:
                    read = csv.reader(csvfile, delimiter=',')
                    for row in read:
                        if (float(row[1]) != 0):
                            score_mat_tmp[int(np.rint(float(row[0]) / 0.0058)), :] = 0
                            score_mat_tmp[int(np.rint(float(row[0]) / 0.0058)), int(
                                np.rint((freq2midi(float(row[1])) - 21) * 4))] = 1
            except FileNotFoundError:
                continue

        score_mat = np.zeros((int(score[-1][0] // 0.02) + 1, 352))
        for i in range(len(score_mat_tmp)):
            score_mat[i] = score_mat_tmp[int(np.round(i * 0.02 / 0.0058))]

    return score_mat


def make_dataset_audio(song_list, label_list, data, dataset_name):

    X = []
    Y = []
    for song in tqdm.tqdm(song_list):

        out = feature_extraction(song)
        score = np.transpose(out[0:4], axes=(2, 1, 0))

        X.append(score)

    if("medleydb" in data ):
        f, v, r = medleydb_preprocessing(song_list)
    else:
        v = None

    for label in tqdm.tqdm(label_list):
        score = label_parser(label, data, v)

        Y.append(score)
    pickle.dump(X, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(Y, open(dataset_name+"_label", 'wb'), pickle.HIGHEST_PROTOCOL)

    print(str(len(X)) + ' files written in ' + dataset_name)


def score_transpose(score, semitone):
    '''
    Tranpose the key of the given score matrix.

    Parameters:
        score: 2D array
            Score matrix.
        semitone: int
             Number of semitone to tranpose up(semitone>0) or down(semitone<0).
    Returns:
        score_tranpose: 2D array
            Tranposed matrix.
    '''
    score_transpose = np.roll(score, shift=semitone, axis=1)
    return score_transpose


def make_dataset_symbolic(songs_list, dataset_name, melody_aug = True, transpose=True):
    '''
    Build pickle dataset given list of midi files.
    Note: melody is shifted by one octave down for half of the songs in the songs_list.

    Parameters:
        songs_list: list
            List of file paths to midi files.
        dataset_name: str
            Output file name.
        tranpose: bool
            whether to augment the datast by transposing the scores.

    '''
    X = []
    switch = 1
    for song in tqdm.tqdm(songs_list):
        if(melody_aug == True):
            switch *= -1
            if (switch == 1):
                score = midi2score(song, melody_mark=True)
            if (switch == -1):
                score = midi2score(song, melody_mark=True, melody_shift=-1)
        else:
            score = midi2score(song, melody_mark=True)

        if transpose:
            timestep_sum = np.sum(score[:, :88], axis=0)

            pitch_range = np.nonzero(timestep_sum)[0]
            max_pitch = pitch_range[-1]
            min_pitch = pitch_range[0]

            min_transposition = 0 - min_pitch
            max_transposition = 87 - max_pitch

            min_transposition = max(-3, min_transposition)
            max_transposition = min(3, max_transposition)

            for semi_tone in range(min_transposition, max_transposition + 1):
                score_transposed = score_transpose(score, semi_tone)
                X.append(score_transposed)

        else:
            X.append(score)

    dataset = (X)
    pickle.dump(dataset, open(dataset_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(str(len(X)) + ' files written in ' + dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type',
                        help='create symbolic or audio data set(default: %(default)s',
                        type=str, default='audio')
    parser.add_argument('-i', '--input_file',
                        help='path to input data(default: %(default)s',
                        type=str, default="../dataset/MIR-1K/Wavfile/")
    parser.add_argument('-ii', '--input_file_label',
                        help='path to label of input data (default: %(default)s',
                        type=str, default="../dataset/MIR-1K/PitchLabel/")
    parser.add_argument('-o', '--output_file',
                        help='name of the output data set(default: %(default)s',
                        type=str, default='data')

    args = parser.parse_args()
    print(args)

    if(args.type == "audio"):
        #load data

        songs = datasets_importer(args.input_file, '.wav')
        songs = np.sort(songs)

        songs_label = datasets_importer(args.input_file_label, 'txt')
        songs_label = np.sort(songs_label)
        
        #make dataset

        make_dataset_audio(songs, songs_label, args.output_file)
    else:
        #load data

        songs = datasets_importer(args.input_file, '.mid')
        songs = np.sort(songs)

        #make dataset

        make_dataset_symbolic(songs, args.output_file, melody_aug = True, transpose=True)

if __name__ == '__main__':
    main()
