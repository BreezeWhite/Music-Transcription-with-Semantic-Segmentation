
import numpy as np 

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

if __name__ == "__main__":
    ltype = BaseLabelType("frame")
    try:
        ltype = BaseLabelType("out_of_mode")
    except ValueError as ex:
        print("Test successful: {}".format(ex))    
