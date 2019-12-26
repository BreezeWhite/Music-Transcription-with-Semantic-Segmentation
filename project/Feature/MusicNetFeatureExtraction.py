import csv

import project.Feature.BaseFeatureExtraction as base
from project.Feature.LabelFormat import LabelFmt

class MusicNetFeatExt(base.BaseFeatExt):
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
