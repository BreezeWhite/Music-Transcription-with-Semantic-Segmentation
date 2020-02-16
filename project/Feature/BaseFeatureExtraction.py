import os
import csv
import glob
import math
import pickle
import librosa
import logging
import numpy as np
from tqdm import trange

from project.Feature.FeatureSecondLayer import process_feature_song_list
from project.Feature.LabelFormat import LabelFmt
from project.configuration import HarmonicNum

class BaseFeatExt:
    def __init__(
            self, 
            wav_path:list, 
            label_path:list, 
            label_ext:str,
            save_path="./train", 
            piece_per_file=40, 
            file_prefix="train",
            harmonic=False
        ):

        self.file_prefix = file_prefix
        self.wav_path = wav_path
        self.label_path = label_path
        self.label_ext = label_ext
        self.num_per_file = piece_per_file
        self.harmonic = harmonic
        self.save_path = save_path
        self.t_unit = 0.02 # Time unit of each frame, in second

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Override this function for your dataset
    def load_label(self, file_path, sample_rate=44100):
        """
        Parameters:
            file_path: str, path to the groundtruth file
            sample_rate: int, sampling rate of an audio
        Return:
            contents: list of LabelFmt, defined in LabelFormat.py 
            frame_num: int, total number of frames of an audio. Each frame is self.t_unit seconds long. 
        For concrete example, please refer to ./MusicNetFeatureExtraction.py
        """
        raise NotImplementedError 

    def process(self):        
        with open(os.path.join(self.save_path, "SongList.csv"), "w", newline='') as csvList:
            writer = csv.DictWriter(csvList, fieldnames=["File name", "id", "Save path"])
            writer.writeheader()
            
            for i, (wav_paths, label_paths)  in enumerate(self.batch_generator()):
                sub_name = os.path.join(self.save_path, ("{}_{}_{}".format(self.file_prefix, self.num_per_file, str(i+1))))

                # Process audio features
                process_feature_song_list(sub_name, wav_paths, harmonic=self.harmonic, num_harmonic=HarmonicNum)

                # Process labels
                self.process_labels(sub_name, label_paths)
                
                # write record to file
                for wav in wav_paths:
                    writer.writerow(
                        {
                            "File name": sub_name,
                            "id": os.path.basename(wav),
                            "Save path": self.save_path
                        }
                    )

    def batch_generator(self):
        # Parse audio files
        all_wavs = []
        for path in self.wav_path:
            all_wavs += glob.glob(os.path.join(path, "*.wav"))
        
        # Parse label files and sort the order of paths in consistent with audio paths
        name_path_map = {}
        for path in self.label_path:
            files = glob.glob(os.path.join(path, "*{}".format(self.label_ext)))
            for ff in files:
                name = os.path.basename(ff).replace(self.label_ext, "")
                name_path_map[name] = ff
        names = [os.path.basename(wav).replace(".wav", "") for wav in all_wavs]
        all_labels = []
        for name in names:
            if name not in name_path_map:
                logging.error("Cannot found corresponding groundtruth file: %s", name)
                continue
            all_labels.append(name_path_map[name])

        # Start to generate batch
        iters = math.ceil(len(all_wavs)/self.num_per_file)
        for i in range(iters):
            print("Iters: {}/{}".format(i+1, iters))
            lower_b = i*self.num_per_file
            upper_b = (i+1)*self.num_per_file
            yield all_wavs[lower_b:upper_b], all_labels[lower_b:upper_b]

    def process_labels(self, sub_name, files):
        '''
        Stored structure:
            labels: Frame x Pitch x Instrument x 2
                Frame: Each frame is <t_unit> second long
                Pitch: Maximum length is 88, equals to the keys of piano
                Instrument: Available types please refer to LabelFormat.py. 
                    It's in Dict type. Only if there exists a instrument will be write to the Dict key-value pair.
                Last dimension: Type of list. [onset_prob, offset_prob]
        '''
        lowest_pitch = librosa.note_to_midi('A0')
        highest_pitch = librosa.note_to_midi('C8')
        pitches = highest_pitch - lowest_pitch + 1
            
        labels = {}
        contains = {} # To summarize a specific instrument is included in pieces
        for idx in trange(len(files), leave=False):
            gt_path = files[idx]
            content, last_sec = self.load_label(gt_path)
            frame_num = int(round(last_sec, 2)/self.t_unit)
            
            label = [{} for i in range(frame_num)]
            for cc in content:
                start_time, end_time, instrument, note, start_beat, end_beat, note_value = cc.get()
                start_f, end_f = int(round(start_time, 2)/self.t_unit), int(round(end_time, 2)/self.t_unit)
                pitch = note-lowest_pitch

                # Put onset probability to the pitch of the instrument
                onsets_v = 1.0
                onsets_len = 2 # 2 frames long
                ii = 0
                for i in range(start_f, end_f):
                    if pitch not in label[i]:
                        label[i][pitch] = {}
                    label[i][pitch][instrument] = [onsets_v, 0] # [onset_prob, offset_prob]
                    ii += 1
                    if ii > onsets_len:
                        onsets_v /= (ii-onsets_len)
                
                # Put offset probability to the pitch of the instrument
                offset_v = 1.0
                offset_len = 4 # 2 frames long
                ii = 0
                for i in range(end_f-1, start_f, -1):
                    label[i][pitch][instrument][1] = offset_v
                    ii += 1
                    if ii >= offset_len:
                        offset_v /= (ii-onsets_len) 

                    # Below are some statistical information generation                    
                    # instrument is contained in pieces
                    if instrument not in contains:
                        contains[instrument] = []
                    name = os.path.basename(gt_path)
                    name = name.rsplit(".", 1)[0]
                    if name not in contains[instrument]:
                        contains[instrument].append(name)
                        contains[instrument].append(idx)
            
            key = os.path.basename(gt_path)
            key = key.replace(self.label_ext, "") # Remove file extension
            labels[key] = label

        pickle.dump(labels, open(sub_name+".pickle", "wb"), pickle.HIGHEST_PROTOCOL)
        return contains

