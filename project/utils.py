import os
import sys
sys.path.append("./")
import json

import h5py
import numpy as np
from scipy.special import expit
from keras.models import model_from_json, model_from_yaml
from keras import layers as L
from keras.utils import multi_gpu_model
from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d

from project.Models.model import seg
from project.Models import model_attn
from project.configuration import get_MusicNet_label_num_mapping, get_instruments_num, MusicNet_Instruments, HarmonicNum


def sigmoid(x):
    #return 1 / (1 + np.exp(-x))
    return expit(x)

def label_conversion(label, tid, 
                     timesteps=128, 
                     ori_feature_size=352, 
                     feature_num=384, 
                     base=88, 
                     mpe=False, 
                     onsets=False,
                     offsets=False):
    '''
    This function is designed for MusicNet.
    Labels formatted with information of different instruments.
    
    See MusicNet/code/ProcessFeature.py for details about the format of labels stored in pickle files.
    '''

    labels = label[tid:(tid+timesteps)]
    if len(labels) < timesteps:
        for _ in range(timesteps-len(labels)):
            labels.append({})
    
    assert(ori_feature_size % base == 0)
    scale = feature_num // base
    spec_inst = MusicNet_Instruments
    inst_num  = get_instruments_num(spec_inst)
    mapping   = get_MusicNet_label_num_mapping(1, spec_inst)
    new_l     = np.zeros((len(labels), ori_feature_size, inst_num+1))
    new_l[:,:,0] = 1

    for t, label in enumerate(labels):
        if len(label.items()) == 0:
            continue

        for pitch, insts in label.items():
            for it in insts:
                if it not in mapping:
                    continue
                    
                ii = mapping[it]
                pr = range(pitch*scale, (pitch+1)*scale)
                #new_l[t, pr, 0] = 0
                if onsets:
                    new_l[t, pr, ii] = insts[it][0]
                elif offsets:
                    new_l[t, pr, ii] = insts[it][1]
                else:
                    new_l[t, pr, ii] = 1

    h = new_l.shape[1]
    p_b = (feature_num-h) // 2
    p_t = feature_num - p_b - new_l.shape[1]
    
    b_shape = (new_l.shape[0], p_b)
    t_shape = (new_l.shape[0], p_t)
    if len(new_l.shape) == 3:
        b_shape += (new_l.shape[2],)
        t_shape += (new_l.shape[2],)
    bottom = np.zeros(b_shape)
    top = np.zeros(t_shape)
    new_l  = np.concatenate([bottom, new_l, top], axis=1)
    
    # This is for single channel output, by merging all the labels into one channel
    if mpe:
        mpe_l = np.nanmax(new_l[:,:,1:], axis=2)
        new_l = np.dstack((new_l[:,:,0], mpe_l))

    new_l[:,:,0] = 1 - np.sum(new_l[:,:,1:], axis=2)
    return new_l


class ModelInfo:
    def __init__(self, model_name="MyModel"):
        self.name = model_name
        self.output_classes = None
        self.label_type = None
        self.timesteps = 256
        self.feature_type = "CFP"
        self.input_channels = [1, 3]
        self.frm_th = 0.5
        self.inst_th = 1.1
        self.onset_th = 6
        self.dura_th = 0
        self.description = None
        self.model_type = None
        
        # Trainig information
        self.dataset = None
        self.epochs = None
        self.steps = None
        self.train_batch_size = None
        self.val_batch_size = None
        self.early_stop = None
        self.loss_function = None

        # Other inner settings
        self._num_gpus = 2
        self._custom_layers = {
            "multihead_attention": model_attn.multihead_attention,
            "Conv2D": L.Conv2D,
            "split_heads_2d": split_heads_2d,
            "local_attention_2d": local_attention_2d,
            "combine_heads_2d": combine_heads_2d
        }

    
    def create_model(self, model_type="attn"):
        self._validate_args()
        self.model_type = model_type
        if model_type == "aspp":
            return seg(
                feature_num=384,
                ch_num=len(self.input_channels),
                timesteps=self.timesteps,
                out_class=self.output_classes,
                multi_grid_layer_n=1,
                multi_grid_n=3
            )
        elif model_type == "attn":
            return model_attn.seg(
                feature_num=384,
                ch_num=len(self.input_channels),
                timesteps=self.timesteps,
                out_class=self.output_classes
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Available: ['attn', 'aspp']")

    def _validate_args(self):
        assert self.output_classes is not None
        assert self.label_type is not None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The given path doesn't exist: {model_path}.")

        # Load model architecture
        self.name = os.path.basename(model_path)
        model = model_from_yaml(
            open(os.path.join(model_path, "arch.yaml")).read(), 
            custom_objects=self._custom_layers
        )

        # Load weights of model
        weight_path = os.path.join(model_path, "weights.h5")
        with h5py.File(weight_path, "r") as w:
            keys = list(w.keys())
            is_para = any(["model" in k for k in keys])

        if is_para:
            para_model = multi_gpu_model(model, gpus=self._num_gpus)
            para_model.load_weights(weight_path)
            model = para_model.layers[-2]
        else:
            model.load_weights(weight_path)

        # Load related configurations
        conf = json.load(open(os.path.join(model_path, "configuration.json")))
        self.name = conf.get("model_name", self.name)
        self.output_classes = conf.get("output_classes", self.output_classes)
        self.label_type = conf.get("label_type", self.label_type)
        self.timesteps = conf.get("timesteps", self.timesteps)
        self.frm_th = conf.get("frame_threshold", self.frm_th)
        self.inst_th = conf.get("instrument_threshold", self.inst_th)
        self.onset_th = conf.get("onset_threshold", self.onset_th)
        self.dura_th = conf.get("duration_threshold", self.dura_th)
        self.description = conf.get("description", self.description)
        self.feature_type = conf.get("feature_type", self.feature_type)
        self.input_channels = conf.get("input_channels", self.input_channels)

        print(f"Model {model_path} loaded")
        return model
    
    def save_model(self, model, save_path):
        path = os.path.join(save_path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save model architecture/weights
        open(os.path.join(path, "arch.yaml"), "w").write(model.to_yaml())
        model.save_weights(os.path.join(path, "weights.h5"))
        
        # Save related configurations
        self.save_configuration(path)
        print(f"Model saved to {save_path}/{self.name}.")

    def save_configuration(self, save_path):
        conf = {
            "model_name": self.name,
            "model_type": self.model_type,
            "output_classes": self.output_classes,
            "label_type": self.label_type,
            "timesteps": self.timesteps,
            "frame_threshold": self.frm_th,
            "instrument_threshold": self.inst_th,
            "onset_threshold": self.onset_th,
            "duration_threshold": self.dura_th,
            "feature_type": self.feature_type,
            "input_channels": self.input_channels,
            "training_settings": {
                "dataset": self.dataset,
                "epochs": self.epochs,
                "steps": self.steps,
                "train_batch_size": self.train_batch_size,
                "val_batch_size": self.val_batch_size,
                "loss_function": self.loss_function,
                "early_stop": self.early_stop
            },
            "description": self._construct_description()
        }
        json.dump(conf, open(os.path.join(save_path, "configuration.json"), "w"), indent=2)
        

    def _construct_description(self):
        return f"""Information about this model
            Model name: {self.name}
            Input feature type: {self.feature_type}
            Input channels: {self.input_channels}
            Timesteps: {self.timesteps}
            Label type: {self.label_type}
            Thresholds:
                Instrument: {self.inst_th}
                Frame: {self.frm_th}
                Onset: {self.onset_th}
                Duration: {self.dura_th}
            Training settings:
                Previously trained on {self.dataset}
                Maximum epochs: {self.epochs}
                Steps per epoch: {self.steps}
                Training batch size: {self.train_batch_size}
                Validation batch size: {self.val_batch_size}
                Loss function type: {self.loss_function}
                Early stopping: {self.early_stop}
        """

    def __repr__(self):
        return self._construct_description()

    def __str__(self):
        return self.description

    @property
    def feature_type(self):
        return self._feature_type

    @feature_type.setter
    def feature_type(self, f_type):
        available = ["CFP", "HCFP"]
        if f_type not in available:
            raise ValueError(f"Invalid feature type: {f_type}. Available: {available}.")
        self._feature_type = f_type

    @property
    def input_channels(self):
        return self._input_channels

    @input_channels.setter
    def input_channels(self, value):
        if not isinstance(value, list):
            value = list(value)

        if len(value) < 5:
            if not all(v in [0, 1, 2, 3] for v in value):
                raise ValueError(f"Invalid channel numbers: {value}. Available: [0, 1, 2, 3].")
        else:
            if len(value)%(HarmonicNum+1) != 0:
                raise ValueError(f"Invalid channel number of harmonic feature: {value}. Length should be multiple of {HarmonicNum+1}.")
        
        self._input_channels = value
        
if __name__ == "__main__":
    minfo = ModelInfo()
    print(minfo)
    #model = minfo.create_model()

