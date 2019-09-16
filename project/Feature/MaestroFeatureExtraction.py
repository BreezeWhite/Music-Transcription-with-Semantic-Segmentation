import pretty_midi
import project.Feature.BaseFeatureExtraction as base
from project.Feature.LabelFormat import LabelFmt 

class MaestroFeatExt(base.BaseFeatExt):
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

        frame_num = last_sec/self.t_unit
        return contents, frame_num



