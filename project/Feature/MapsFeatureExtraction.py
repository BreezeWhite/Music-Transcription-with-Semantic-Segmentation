
import project.Feature.BaseFeatureExtraction as base
#import project.Feature.BaseFeatureExtraciton.BaseFeatExt
from project.Feature.LabelFormat import LabelFmt 

class MapsFeatExt(base.BaseFeatExt):
    # Override
    def load_label(self, file_path, sample_rate=44100):
        with open(file_path, "r") as ll_file:
            lines = ll_file.readlines()

        content = []
        last_sec = 0
        for i in range(1, len(lines)):
            if lines[i].strip() == "":
                break
            onset, offset, note = lines[i].split("\t")
            onset, offset, note = float(onset), float(offset), int(note[:note.find("\n")])
            content.append(LabelFmt(onset, offset, note)) 
            last_sec = max(last_sec, offset)
        
        frame_num = last_sec/self.t_unit
        return content, frame_num
