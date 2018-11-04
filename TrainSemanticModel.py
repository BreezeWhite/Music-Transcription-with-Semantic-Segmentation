import os
import csv
import glob
import argparse

from project.utils import load_model, save_model
from project.configuration import Harmonic_Num
from project.model import seg, sparse_loss
from project.train import train_audio

from keras import callbacks 

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--MAPS-feature-path',
                        help='Path to pre-proccessed MAPS features',
                        type=str)
                        
    musicnet_parser = parser.add_argument_group("MusicNet parameters")
    musicnet_parser.add_argument('--MusicNet-feature-path',
                                 help='Path to pre-proccessed MusicNet features',
                                 type=str)
    musicnet_parser.add_argument('--no-harmonic',
                                 help="Wether to use HCFP feature to train the model",
                                 action='store_true')
    musicnet_parser.add_argument('--mpe-only',
                                 help='Wether to train on MPE task (pitches only, without individual instrument \
                                 prediction',
                                 action='store_true')
    
    parser.add_argument('-nd', '--num-datasets',
                        help='Number of train_30.pickle files to load (default: %(default)d)',
                        type=int, default=3)
    parser.add_argument('-c', '--channels',
                        help='Channels to use for training (default: %(default)s)',
                        type=int, nargs='+', default=[1, 3]) # Candidates are: (0: Z), (1: Spec), (2: GCoS), 
                                                             #                 (3: Ceps), for MAPS


    #arguments for training
    parser.add_argument('-t', '--model_type',
                        help='model type: seg or pnn (default: %(default)s)',
                        type=str, default='seg')
    parser.add_argument('-ms', '--model_path_symbolic',
                        help='path to symbolic model (default: %(default)s)',
                        type=str, default='model_symbolic')
    parser.add_argument('--use-ram',
                        help='If your ram is big enough for the entire dataset, it is recommand to turn this on, \
                        or the training progress would be very slow',
                        action='store_true')
    parser.add_argument('-w', '--window_width',
                        help='width of the input feature (default: %(default)d)',
                        type=int, default=128)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size during training (default: %(default)d)',
                        type=int, default=12)
    parser.add_argument('-e', '--epoch',
                        help='number of epoch (default: %(default)s)',
                        type=int, default=5)
    parser.add_argument('-n', '--steps',
                        help='number of step per epoch (default: %(default)d)',
                        type=int, default=6000)
    parser.add_argument('--input_model',
                        help='Continue to train on a pre-trained model',
                        type=str)
    parser.add_argument('-o', '--output_model_name',
                        help='name of the output model (default: %(default)s)',
                        type=str, default="onsets_model_ + <list_of_channel_used>")


    args = parser.parse_args()
    print("\n", args, "\n")
    

    # Arguments setting
    TIMESTEPS = args.window_width
    if " + " in args.output_model_name:
        args.output_model_name = args.output_model_name[0:13] + str(args.channel)
    
    ch_num = len(args.channels)
    if args.MusicNet_feature_path is not None:
        base_path    = args.MusicNet_feature_path
        dataset_type = "MusicNet"
        # Input parameters
        if args.no_harmonic == True:
            ch_num = 2
            args.channels = [0, 6] # Spec. and Ceps. channel
            feature_type  = "CFP"
        else:
            ch_num = Harmonic_Num * 2
            args.channels = [i for i in range(ch_num)] # Including harmonic channels
            feature_type  = "HCFP"
        # Output parameters
        if args.mpe_only:
            out_classes = 2
        else:
            out_classes = 12
            
    elif args.MAPS_feature_path is not None:
        base_path = args.MAPS_feature_path
        out_classes = 2
        dataset_type = "MAPS"
        feature_type = "CFP"
        args.no_harmonic = True
    else:
        assert(False), "Please at least assign one of the flags: --MAPS-feature-path or --MusicNet-feature-path"
        
    distinct_file = set()
    with open(os.path.join(base_path, "SongList.csv"), newline='') as config:
        reader = csv.DictReader(config)
        for row in reader:
            distinct_file.add(row["File name"])
    dataset_path = [ff for ff in distinct_file][0:args.num_datasets]
    label_path   = [i+"_label.pickle" for i in dataset_path]
    dataset_path = [i+".hdf" for i in dataset_path]
    print("Datasets chosen: ", dataset_path)

    dataset_path = [os.path.join(base_path, dp) for dp in dataset_path] 
    label_path   = [os.path.join(base_path, lp) for lp in label_path]
    
    
    # Load/create model
    if args.input_model is not None:
        model = load_model(args.input_model)
    else: 
        model = seg(multi_grid_layer_n=1, feature_num=384, input_channel=ch_num, timesteps=TIMESTEPS,
                    out_class=out_classes)
        model.compile(optimizer="adam", loss={'prediction': sparse_loss}, metrics=['accuracy'])
    
    # create callbacks
    path = os.path.join("model", args.output_model_name)
    earlystop   = callbacks.EarlyStopping(monitor="val_acc", patience=7)
    checkpoint  = callbacks.ModelCheckpoint(path+"/weights.h5", monitor="val_acc", 
                                            save_best_only=True, save_weights_only=True)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join("tensorboard", args.output_model_name),
                                        write_images=True)
    callback_list = [checkpoint, earlystop, tensorboard]

    # Save model and configurations
    if not os.path.exists(path):
        os.makedirs(path)
    save_model(model, path, feature_type=feature_type, input_channels=args.channels, output_classes=out_classes)
    
    # Train model
    train_audio(model, 
                timesteps    = TIMESTEPS, 
                dataset_path = dataset_path, 
                label_path   = label_path, 
                callbacks    = callback_list,
                epoch        = args.epoch, 
                steps        = args.steps, 
                batch_size   = args.batch_size_train, 
                channels     = args.channels,
                dataset_type = dataset_type,
                use_ram      = args.use_ram,
                mpe_only     = args.mpe_only)

    


if __name__ == '__main__':
    main()
