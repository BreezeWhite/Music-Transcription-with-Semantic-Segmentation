import numpy as np
import tqdm

from project.utils import *

def generation_prog(model,
                    score_48,
                    score_12,
                    time_index,
                    timesteps,
                    batch_size,
                    channel):

    feature_48 = score_48[time_index:time_index + batch_size, :, :]
    feature_48 = np.reshape(feature_48, (batch_size, timesteps, 384, channel))
    feature_12 = score_12[time_index:time_index + batch_size, :, :]
    feature_12 = np.reshape(feature_12, (batch_size, timesteps, 128, channel))

    input_features = {'input_score_48': feature_48,
                      'input_score_12': feature_12
                      }

    probas = model.predict(input_features, batch_size=batch_size)

    return probas


def inference(feature,
              model,
              timestep=TIMESTEP,
              batch_size=10,
              feature_num_12=128,
              feature_num_48=384,
              channel=1,
              threshold = 0.4,
              isMPE = False, 
              original_v = False,
              instruments=1,
              keep_progress=True):
    
    # Initialize varables

    #f_12 = note_res_downsampling(feature)
    f_12   = np.zeros((feature.shape[0], feature_num_12, channel))
    f_12_p = padding(f_12, feature_num_12, timestep)
    f_12_s = np.zeros((f_12_p.shape[0], timestep, f_12_p.shape[1], f_12_p.shape[2]))
    del f_12

    f_48_p, p_t, p_b = padding(feature, feature_num_48, timestep, dimension=True)
    f_48_s = np.zeros((f_48_p.shape[0], timestep, f_48_p.shape[1], f_48_p.shape[2]))

    for i in range(len(f_12_s) - timestep):
        f_12_s[i] = f_12_p[i:i + timestep]
        f_48_s[i] = f_48_p[i:i + timestep]

    extract_result_seg = np.zeros(f_48_s.shape[:-1] + (instruments+1,))
    extract_result_seg_flatten = np.zeros(f_48_p.shape[:-1] + (instruments+1,))
    del f_48_p, f_12_p
    

    # Start to generate result
    iter_num = int(np.ceil(((len(f_12_s) - timestep) / batch_size)))

    for i in tqdm.tqdm(range(iter_num), desc='Fragments', leave=keep_progress): 
        time_index = i * batch_size
        p = generation_prog(model, f_48_s, f_12_s, time_index=time_index, timesteps=timestep,
                            batch_size=batch_size, channel=channel)
        p = sigmoid(p)
        extract_result_seg[time_index:time_index + batch_size] = p
    
    # Post process
    for i in range(len(f_12_s) - timestep):
        extract_result_seg_flatten[i:i + timestep] += extract_result_seg[i]
    del f_48_s, f_12_s

    # Filter values by threshold
    extract_result_seg = extract_result_seg_flatten[timestep:-timestep, p_t:-p_b, 1:]
    del extract_result_seg_flatten
    
    avg = 0
    if not isMPE:
        for i, step in enumerate(extract_result_seg):
            maximum = np.sort(step)[-1]
            avg += maximum
            extract_result_seg[i][extract_result_seg[i] < maximum] = 0

        avg /= extract_result_seg.shape[0]
        extract_result_seg[extract_result_seg < avg] = 0
        extract_result_seg[extract_result_seg > avg] = 1
    else: 
        # Below is suitable for MPE problems (e.g. piano roll)
        extract_result_seg /= feature_num_12 # Maximum value after add would be feature_num_12(batch size or 128)
        if not original_v:
            extract_result_seg = np.where(extract_result_seg>threshold, 1, 0) 

    return extract_result_seg
