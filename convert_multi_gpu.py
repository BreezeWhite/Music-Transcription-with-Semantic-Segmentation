
import os
from project.utils import *


model_name = "./model/Maestro-Attn-W4.2"
save_name = "./Maestro-Attn-W4.2-SingleGPU"
model = load_model(model_name)
m_info = model_info(model_name)

if not os.path.exists(save_name):
    os.makedirs(save_name)

save_model(model, save_name, *m_info)
model.save_weights(save_name + "/weights.h5")

