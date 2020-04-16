import os
import csv
import glob
import math
import pickle
import librosa
import logging
import numpy as np
import pretty_midi
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
                #process_feature_song_list(sub_name, wav_paths, harmonic=self.harmonic, num_harmonic=HarmonicNum)

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

        # Parse label files
        name_path_map = {}
        for path in self.label_path:
            files = glob.glob(os.path.join(path, "*{}".format(self.label_ext)))
            for ff in files:
                name = self.name_transform(os.path.basename(ff))
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

    def name_transform(self, original):
        """
        Transform a ground truth file name to the correponding audio name (without extension).
        """
        return original.replace(self.label_ext, "")


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
            
            key = self.name_transform(os.path.basename(gt_path))
            labels[key] = label

        pickle.dump(labels, open(sub_name+".pickle", "wb"), pickle.HIGHEST_PROTOCOL)
        return contains



class MaestroFeatExt(BaseFeatExt):
    # Override
    def load_label(self, file_path, **kwargs):
        midi = pretty_midi.PrettyMIDI(file_path)
        inst = midi.instruments[0] # Assumed there only exits piano

        contents = []
        last_sec = 0
        for note in inst.notes:
            onset = note.start
            offset = note.end
            pitch = note.pitch
            contents.append(LabelFmt(onset, offset, pitch))
            last_sec = max(last_sec, offset)

        return contents, last_sec


class MapsFeatExt(BaseFeatExt):
    # Override
    def load_label_midi(self, file_path, **kwargs):
        midi = pretty_midi.PrettyMIDI(file_path)
        inst = midi.instruments[0]

        content = []
        last_sec = 0
        for note in inst.notes:
            onset = note.start
            offset = note.end
            pitch = note.pitch
            content.append(LabelFmt(onset, offset, pitch))
            last_sec = max(last_sec, offset)

        return content, last_sec


    def load_label(self, file_path, sample_rate=44100):
        with open(file_path, "r") as ll_file:
            lines = ll_file.readlines()

        content = []
        last_sec = 0
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                continue
            onset, offset, note = lines[i].split("\t")
            onset, offset, note = float(onset), float(offset), int(note.strip())
            content.append(LabelFmt(onset, offset, note)) 
            last_sec = max(last_sec, offset)
        
        return content, last_sec


class MusicNetFeatExt(BaseFeatExt):
    # Override
    def load_label(self, file_path, sample_rate=44100):
        content = []
        last_sec = 0
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            max_sample = 0
            for label in reader:
                start_time = float(label['start_time'])/sample_rate
                end_time   = float(label['end_time'])/sample_rate
                instrument = int(label['instrument'])
                note       = int(label['note'])
                start_beat = float(label['start_beat'])
                end_beat   = float(label['end_beat'])
                note_value = label['note_value']

                cc = LabelFmt(start_time, end_time, note, instrument, start_beat, end_beat, note_value)
                content.append(cc)
                last_sec = max(last_sec, end_time)

        return content, last_sec


class SuFeatExt(BaseFeatExt):
    # Override
    def load_label(self, file_path, sample_rate=44100):
        midi = pretty_midi.PrettyMIDI(file_path)
        content = []
        last_sec = 0
        for inst in midi.instruments:
            inst_num = inst.program
            for note in inst.notes:
                cc = LabelFmt(note.start, note.end, note.pitch, inst_num)
                content.append(cc)
                last_sec = max(last_sec, note.end)
        return content, last_sec


class RhythmFeatExt(SuFeatExt):
    # Override
    def name_transform(self, original):
        new_name = original.replace("align_mid", "ytd_audio")
        return new_name.replace(".mid", ".mp3")

