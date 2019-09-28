import tensorflowjs as tfjs
from project.utils import load_model

model_path = "model/maestro"
save_path = "model/maestro_tfjs"

model = load_model(model_path)
tfjs.converters.save_keras_model(model, save_path)

