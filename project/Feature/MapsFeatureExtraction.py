
import pretty_midi
import project.Feature.BaseFeatureExtraction as base
from project.Feature.LabelFormat import LabelFmt 

class MapsFeatExt(base.BaseFeatExt):
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
