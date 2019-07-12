import pickle
import os
import numpy as np

from keras.utils.np_utils import to_categorical

from project.utils import *
from project.configuration import get_instruments_num, MusicNet_Instruments


def generator_audio2(batch_size,
                     timesteps,
                     data_path,
                     label,
                     phase='train',
                     percentage_train=0.8,
                     channels=None,
                     dataset_type="MusicNet",
                     use_ram=False,
                     mpe_only=False):

    Y = label[:]
    data = load_hdf(data_path[0], inplace=use_ram)
    
    # Small hack for maestro dataset
    val_data = load_hdf(data_path[1])
    len_train = len(data)
    len_val = len(val_data)

    for i in range(len_val):
        data.append(val_data[i])
    # end of hack

    
    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(data) * percentage_train))
        chorale_indices = np.arange(len_train) # hack for maestro
    if phase == 'test':
        chorale_indices = np.arange(int(len(data) * percentage_train), len(data))
        chorale_indices = np.arange(len_train, len_train+len_val) # hack for maestro
    if phase == 'all':
        chorale_indices = np.arange(int(len(data)))

    for a in range(len(label)):
        if a in chorale_indices:
            new_y = np.array(Y[a])
            Y[a] = padding(new_y, 384, timesteps, muti_instrument=True)

    features_48 = []
    features_12 = []
    labels = []

    batch = 0
    while True:
        chorale_index = np.random.choice(chorale_indices) # Select piece
        chorale_length = min(len(Y[chorale_index]), len(data[chorale_index]))

        time_index = np.random.randint(0, chorale_length - timesteps)

        feature_48 = augment_hdf(data, chorale_index, time_index, timesteps, channels)
        feature_12 = np.zeros((timesteps, 128, 1))

        label = Y[chorale_index][time_index: time_index+timesteps]
        if dataset_type=="MusicNet":
            label = label_conversion(label, 384, timesteps, mpe=mpe_only)#, onsets=True)
        else:
            label = to_categorical(label, num_classes=2)

        features_48.append(feature_48)
        features_12.append(feature_12)
        labels.append(label)

        batch += 1

        if batch == batch_size:
            next_element = (
                np.array(features_48, dtype=np.float32),
                np.array(features_12, dtype=np.float32),
                np.array(labels, dtype=np.float32)
            )

            yield next_element

            batch = 0
            features_48 = []
            features_12 = []
            labels = []


def generator_audio(batch_size, 
                    timesteps, 
                    dataset, 
                    label,
                    phase='train', 
                    percentage_train=0.8,
                    constraint=False,
                    channels=[0],
                    spec_inst=None):
    
    X_48 = dataset[:]
    Y    = label[:]
    

    instruments = get_instruments_num(spec_inst)
    
    
    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X_48) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X_48) * percentage_train), len(X_48))
    if phase == 'all':
        chorale_indices = np.arange(int(len(X_48)))

    for a in range(len(X_48)):
        if (a in chorale_indices):
            new_x = np.array(X_48[a][:, :, channels])
            if new_x.shape[2] == 1:
                new_x = new_x.squeeze(axis=2)
    
            new_x_48 = padding(new_x, 384, timesteps)
            X_48[a] = new_x_48
            
            new_y = np.array(Y[a]) # Dim: frames x roll
            Y[a] = padding(new_y, 384, timesteps, muti_instrument=(instruments>1))

    features_48 = []
    features_12 = []
    labels = []

    batch = 0
    
    while True:
        chorale_index = np.random.choice(chorale_indices)
        #chorale = np.array(X_48[chorale_index])
        chorale_length = len(Y[chorale_index])#chorale)
        time_index = np.random.randint(0, chorale_length - timesteps)
        
        #print(chorale_length, len(chorale))
        feature_48 = (X_48[chorale_index][time_index: time_index + timesteps])
        feature_48 = np.reshape(feature_48, (timesteps, 384, len(channels)))

        #feature_12 = (X_12[chorale_index][time_index: time_index + timesteps])
        #feature_12 = np.reshape(feature_12, (timesteps, 128, 1))
        feature_12 = np.zeros((timesteps, 128, 1))
        
        label = Y[chorale_index][time_index: time_index + timesteps]
        #print(chorale_length, len(Y[chorale_index]))

        if instruments>1:
            label = label_conversion(label, 384, timesteps, spec_inst=spec_inst)
        else:
            label = to_categorical(label, num_classes=(instruments+1))

        features_48.append(feature_48)
        features_12.append(feature_12)
        labels.append(label)

        batch += 1

        del feature_12, feature_48, label#, chorale

        # if there is a full batch
        if batch == batch_size:
            next_element = (
                np.array(features_48, dtype=np.float32),
                np.array(features_12, dtype=np.float32),
                np.array(labels, dtype=np.float32)
            )

            yield next_element

            del features_48, features_12, labels

            batch = 0
            
            features_48 = []
            features_12 = []
            labels = []


def train_audio(model, 
                timesteps, 
                dataset_path, 
                label_path, 
                callbacks=None,
                epoch=5, 
                steps=6000, 
                batch_size=12, 
                channels=[1, 3],
                dataset_type="MusicNet",
                use_ram=False,
                mpe_only=False):
    
    label = load_data(label_path)
    
    #"""
    # small hack for maestro dataset
    val_path = "/media/whitebreeze/本機磁碟/maestro-v1.0.0/feature_val"
    
    distinct_file = set()
    with open(os.path.join(val_path, "SongList.csv"), newline='') as config:
        reader = csv.DictReader(config)
        for row in reader:
            distinct_file.add(row["File name"])
    valf_path = [ff for ff in distinct_file]
    val_label_path   = [i+"_label.pickle" for i in valf_path]
    valf_path = [i+".hdf" for i in valf_path]
    valf_path = [os.path.join(val_path, dp) for dp in valf_path] 
    val_label_path   = [os.path.join(val_path, lp) for lp in val_label_path]
    val_label = load_data(val_label_path)

    dataset_path = [dataset_path, valf_path]
    label = np.concatenate([label, val_label])
    # end hack
    #"""

    generator_train = (({'input_score_48': features_48,
                         'input_score_12': features_12
                         },
                        {'prediction': labels}
                        )
                       for (features_48,
                            features_12,
                            labels
                            ) in generator_audio2(batch_size, timesteps, dataset_path, label, 
                                                  channels=channels, dataset_type=dataset_type, use_ram=use_ram,
                                                  mpe_only=mpe_only))
    
    generator_val = (({'input_score_48': features_48,
                       'input_score_12': features_12
                       },
                      {'prediction': labels}
                      )
                     for (features_48,
                          features_12,
                          labels
                          ) in generator_audio2(batch_size, timesteps, dataset_path, label,
                                                phase='test', channels=channels, dataset_type=dataset_type, use_ram=use_ram,
                                                mpe_only=mpe_only))

    model.fit_generator(generator_train, steps_per_epoch=steps,
                        epochs=epoch, verbose=1, validation_data=generator_val,
                        validation_steps=200,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        max_queue_size=100,
                        workers=1,
                        )


    return model
