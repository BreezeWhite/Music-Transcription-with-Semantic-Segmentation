import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np 

import sys
sys.path.append("./")

from project.utils import label_conversion


class BaseLabelType:
    def __init__(self,
                 mode:str,
                 timesteps=128):
        self.mode = mode
        self.timesteps = timesteps
        self.l_conv = lambda label, tid, **kwargs: label_conversion(label, tid, timesteps=timesteps, **kwargs)

        if mode == "frame":
            self.conversion_func = self.get_frame
            self.out_classes = 2
        elif mode == "frame_onset":
            self.conversion_func = self.get_frame_onset
            self.out_classes = 3
        elif mode == "frame_onset_offset":
            self.conversion_func = self.get_frame_onset_offset
            self.out_classes = 4
        elif self.customized_mode(mode):
            pass
        else:
            raise ValueError("Available mode: 'frame', 'frame_onset', 'frame_onset_offset'. Provided: {}".format(mode))
    
    # Override this function if you have implemented a new mode
    def customized_mode(self, mode)->bool:
        self.convserion_func = None
        self.out_classes = None
        return False

    def get_conversion_func(self):
        return self.conversion_func

    def get_out_classes(self)->int:
        return self.out_classes

    def get_frame(self, label, tid)->np.ndarray:
        return self.l_conv(label, tid, mpe=True)

    def get_frame_onset(self, label, tid)->np.ndarray:
        frame = self.get_frame(label, tid)
        onset = self.l_conv(label, tid, onsets=True, mpe=True)[:,:,1]

        frame[:,:,1] -= onset
        frm_on = np.dstack([frame, onset])
        frm_on[:,:,0] = 1-np.sum(frm_on[:,:,1:], axis=2)

        return frm_on

    def get_frame_onset_offset(self, label, tid)->np.ndarray:
        frm_on = self.get_frame_onset(label, tid)
        offset = self.l_conv(label, tid, offsets=True)

        tmp = frm_on[:,:,1] - frm_on[:,:,2] - offset
        tmp[tmp>0] = 0
        offset += tmp
        frm_on[:,:,1] = frm_on[:,:,1] - frm_on[:,:,2] - offset

        frm_on_off = np.dstack([frm_on, offset])
        frm_on_off[:,:,0] = 1-np.sum(frm_on_off[:,:,1:], axis=2)

        return frm_on_off


class MusicNetLabelType(BaseLabelType):
    def customized_mode(self, mode):
        if mode == "multi_instrument_frame":
            self.conversion_func = self.multi_inst_frm
            self.out_classes = 12
        elif mode == "multi_instrument_note":
            self.conversion_func = self.multi_inst_note
            self.out_classes = 23
        else:
            self.conversion_func = None
            self.out_classes = none
            return False

        return True

    def multi_inst_frm(self, label, tid):
        return self.l_conv(label, tid)

    def multi_inst_note(self, label, tid):
        onsets = self.l_conv(label, tid, onsets=True)
        dura = self.l_conv(label, tid) - onsets
        out = np.zeros(onsets.shape[:-1]+(23,))

        for i in range(11):
            out[:,:,i*2+2] = onsets[:,:,i+1]
            out[:,:,i*2+1] = dura[:,:,i+1]
        out[:,:,0] = 1 - np.sum(out[:,:,1:], axis=2)

        return out

if __name__ == "__main__":
    ltype = BaseLabelType("frame")
    try:
        ltype = BaseLabelType("out_of_mode")
    except ValueError as ex:
        print("Test successful: {}".format(ex))    

    label = pickle.load(open("/data/MusicNet/train_feature/train_20_14.pickle", "rb"))
    key = list(label.keys())
    val = label[key[3]]
    ltype = MusicNetLabelType("multi_instrument_note", timesteps=len(val))

    roll = ltype.multi_inst_note(val, 0)
    for i in range(11):
        plt.imshow(roll[:,:,i*2+1].transpose(), origin="lower", aspect="auto")
        plt.savefig("{}_onset.png".format(i), dpi=250)
        plt.imshow(roll[:,:,i*2+2].transpose(), origin="lower", aspect="auto")
        plt.savefig("{}_dura.png".format(i), dpi=250)

