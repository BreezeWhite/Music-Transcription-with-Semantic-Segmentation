
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from project.utils import *
from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d
from keras.models import model_from_yaml


model_name = "./model/MusicNet-Attn-W4.2"
save_name = "./MusicNet-Attn-W4.2-SingleGPU"

model = load_model(model_name)
m_info = model_info(model_name)

if not os.path.exists(save_name):
    os.makedirs(save_name)

save_model(model, save_name, *m_info)
model.save_weights(save_name + "/weights.h5")

