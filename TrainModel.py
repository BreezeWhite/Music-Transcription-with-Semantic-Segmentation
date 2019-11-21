
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import argparse

from project.LabelType import BaseLabelType, MusicNetLabelType
from project.utils import load_model, save_model, model_info
from project.configuration import HarmonicNum
from project.Models.model import seg, sparse_loss, smooth_loss, mctl_loss
from project.Models import model_attn
from project.Dataflow import DataFlows

from keras import callbacks
from keras.utils import multi_gpu_model
import tensorflow as tf

dataset_paths = {
    "Maestro":  "/data/Maestro",
    "MusicNet": "/data/MusicNet",
    "Maps":     "/data/Maps"
}

dataflow_cls = {
    "Maestro":  DataFlows.MaestroDataflow,
    "MusicNet": DataFlows.MusicNetDataflow,
    "Maps":     DataFlows.MapsDataflow
}

default_model_path = "./model"


def train(
        model, 
        generator_train, 
        generator_val,
        epoch=1,
        callbacks=None, 
        steps=6000, 
        v_steps=3000
    ):

    model.fit_generator(
        generator_train, 
        validation_data=generator_val,
        epochs=epoch,
        steps_per_epoch=steps,
        validation_steps=v_steps,
        callbacks=callbacks,
        max_queue_size=100,
        use_multiprocessing=False)
    #    workers=1
    #)

    return model

def main(args):
    if args.dataset not in dataflow_cls:
        raise TypeError
    
    # Hyper parameters that will be stored for future reuse
    hparams = {}

    # Parameters that will be passed to dataflow
    df_params = {}
    
    # Handling root path to the dataset
    d_path = dataset_paths[args.dataset]
    if args.dataset_path is not None:
        assert(os.path.isdir(args.dataset_path))
        d_path = args.dataset_path
    
    # Number of channels that model need to know about
    ch_num = len(args.channels)
    channels = args.channels
    
    # Type of feature to use
    feature_type = "CFP"
    
    # Output model name
    out_model_name = args.output_model_name
    
    # Feature length on time dimension
    timesteps = args.timesteps

    # Label type
    mode = "frame_onset"
    l_type = BaseLabelType(mode, timesteps=timesteps)
    mode = "multi_instrument_note"
    l_type = MusicNetLabelType(mode, timesteps=timesteps)

    # Number of output classes
    out_classes = l_type.get_out_classes()

    # Continue to train on a pre-trained model
    if args.input_model is not None:
        # load configuration of previous training
        feature_type, channels, out_classes, timesteps = model_info(args.input_model)
        ch_num = len(channels)
    else:
        if args.dataset == "MusicNet":
            # Sepcial settings for MusicNet that has multiple instruments presented
            if args.use_harmonic:
                ch_num = HarmonicNum * 2
                channels = [i for i in range(ch_num)]
                feature_type = "HCFP"

    df_params["b_sz"]      = args.train_batch_size
    df_params["phase"]     = "train"
    df_params["use_ram"]   = args.use_ram
    df_params["channels"]  = channels
    df_params["timesteps"] = timesteps
    df_params["out_classes"]  = out_classes
    df_params["dataset_path"] = d_path
    df_params["label_conversion_func"] = l_type.get_conversion_func()

    print("Loading training data")
    df_cls = dataflow_cls[args.dataset]
    train_df = df_cls(**df_params)

    print("Loading validation data")
    df_params["b_sz"]  = args.val_batch_size
    df_params["phase"] = "val"
    val_df = df_cls(**df_params)

    hparams["channels"]       = channels
    hparams["timesteps"]      = timesteps
    hparams["feature_type"]   = feature_type
    hparams["output_classes"] = out_classes
    
    print("Creating/loading model")
    # Create model
    if args.input_model is not None:
        model = load_model(args.input_model)
    else:
        # Create new model
        #model = seg(multi_grid_layer_n=1, feature_num=384, input_channel=ch_num, timesteps=timesteps,
        #            out_class=out_classes)
        model = model_attn.seg(feature_num=384, input_channel=ch_num, timesteps=timesteps,
                               out_class=out_classes)

    # Save model and configurations
    out_model_name = os.path.join(default_model_path, out_model_name)
    if not os.path.exists(out_model_name):
        os.makedirs(out_model_name)
    save_model(model, out_model_name, **hparams)

    # Weighted loss
    weight = [1, 1, 2.5] # Piano-only note mode
    weight = None # Frame mode
    # Multi-instrument note mode
    weight = [1 for i in range(23)]
    for i in range(11):
        weight[i*2+1] = 2.5
    ##
    if weight is not None:
        assert(len(weight)==out_classes),"Weight length: {}, out classes: {}".format(len(weight), out_classes)
    loss_func = lambda label,pred: mctl_loss(label, pred, weight=weight)
    
    # Use multi-gpu to train the model
    if True:
        para_model = multi_gpu_model(model, gpus=2, cpu_merge=False)
        para_model.compile(optimizer="adam", loss={'prediction': loss_func}, metrics=['accuracy'])
        model = para_model
    else:
        model.compile(optimizer="adam", loss={'prediction': loss_func}, metrics=['accuracy'])

    # create callbacks
    earlystop   = callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop)
    checkpoint  = callbacks.ModelCheckpoint(os.path.join(out_model_name, "weights.h5"), 
                                            monitor="val_loss", save_best_only=False, save_weights_only=True)
    tensorboard = callbacks.TensorBoard(log_dir=os.path.join("tensorboard", args.output_model_name),
                                        write_images=True)
    callback_list = [checkpoint, earlystop, tensorboard]
    
    print("Start training")
    # Start training
    train(model, train_df, val_df,
          epoch     = args.epoch,
          callbacks = callback_list,
          steps     = args.steps,
          v_steps   = args.val_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Frame-level polyphonic music transcription project done by MCTLab, IIS Sinica.")
    
    parser.add_argument("dataset", help="One of the Maestro, MusicNet, or Maps",
                        type=str)
    parser.add_argument("output_model_name", help="Name for trained model. If --input-model is given, \
                        then this flag has no effect.", type=str)
    parser.add_argument("-d", "--dataset-path", help="Path to the root of the dataset that has preprocessed feature",
                        type=str)
    parser.add_argument("--use-harmonic", help="Wether to use HCFP feature to train the model",
                        action="store_true")
    parser.add_argument("--multi-instruments", help="Train on transcribing the note played with different instruments",
                        action="store_true")
    # Channel types
    #   0: Z
    #   1: Spec
    #   2: GCoS
    #   3: Ceps
    parser.add_argument("-c", "--channels", help="Use specific channels of feature to train (default: %(default)d)",
                        type=int, nargs="+", default=[1, 3]) 
    parser.add_argument("--use-ram", help="Wether to load the whole dataset into ram",
                        action="store_true")
    parser.add_argument("-t", "--timesteps", help="Time width for each input feature (default: %(default)d)",
                        type=int, default=128)
    
    # Arguments about the training progress
    parser.add_argument("-e", "--epoch", help="Number of epochs to train (default: %(default)d)",
                        type=int, default=10)
    parser.add_argument("-s", "--steps", help="Training steps for each epoch (default: %(default)d)",
                        type=int, default=2000)
    parser.add_argument("-vs", "--val-steps", help="Validation steps (default: %(default)d)",
                        type=int, default=500)
    parser.add_argument("-i", "--input-model", help="If given, then will continue to train on a pre-trained model")
    parser.add_argument("-b", "--train-batch-size", help="Batch size for training phase (default: %(default)d)",
                        type=int, default=8)
    parser.add_argument("-vb", "--val-batch-size", help="Batch size for validation phase (default: %(default)d)",
                        type=int, default=16)
    parser.add_argument("--early-stop", help="Early stop the training after given # epochs",
                        type=int, default=4)

    args = parser.parse_args()
    print(args)
    main(args)



