import pretty_midi
from project.Feature.BaseFeatureExtraction import BaseFeatExt
from project.Feature.LabelFormat import LabelFmt


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
