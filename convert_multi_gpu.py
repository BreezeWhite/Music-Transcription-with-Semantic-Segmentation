
import os
from project.utils import *
from tensor2tensor.layers.common_attention import local_attention_2d, split_heads_2d, combine_heads_2d
from keras.models import model_from_yaml

model_name = "./model/Maestro-Attn-V4.1"
save_name = "./Maestro-Attn-V4.1-SingleGPU"
#model = load_model(model_name)
m_info = model_info(model_name)

if not os.path.exists(save_name):
    os.makedirs(save_name)

#save_model(model, save_name, *m_info)
#full_path = os.path.join(model_path, "arch.yaml")
#model = model_from_yaml(open(full_path).read())
custom_layers = {
    "multihead_attention": multihead_attention,
    "Conv2D": L.Conv2D,
    "split_heads_2d": split_heads_2d,
    "local_attention_2d": local_attention_2d,
    "combine_heads_2d": combine_heads_2d
}
model = model_from_yaml(open(os.path.join(model_name, "arch.yaml")).read(), custom_objects=custom_layers)
para_model = multi_gpu_model(model, gpus=2)

full_path = os.path.join(model_name, "weights.h5")
para_model.load_weights(full_path)
model = para_model.layers[-2]
model_arch = model.to_yaml()
open(os.path.join(save_name, "arch.yaml"), "w").write(model_arch)

model.save_weights(save_name + "/weights.h5")

