
# Music Transcription with Semantic Model

## About

This is a AMT (Automatic  Music Transcription) project, combined with state-of-the-art image semantic segmentation neural network model. 

The dataset used is MAPS and MusicNet, which the first one is a solo-piano performance collection, and the second is a multi-instrument performance collection.  On both dataset, we achieved the state-of-the-art results on MPE (Multi-Pitch Estimation) case frame-wisely, which on **MAPS** we achieved **F-score 86.73%**, and on **MusicNet** we achieved **F-score 73.70%**.

This work was done based on our prior work of [repo1](https://github.com/BreezeWhite/CFP_NeuralNetwork), [repo2](https://github.com/s603122001/Vocal-Melody-Extraction). For more about our works, please meet our [website](https://sites.google.com/view/mctl/home).

For whom would interested in more technical details, the original paper is [here](https://ieeexplore.ieee.org/abstract/document/8682605).

### Colab
We have also provide a [colab](https://colab.research.google.com/drive/1U6FxnQoeENZRcTRAg4DzjKYM7ZNPGTks?hl=en#offline=true&sandboxMode=true) for direct testing. Just run through each block and upload your .mp3 piano recording, then enjoy.


## Table of Contents

* [About](#About)
* [Overview](#overview)
* [Install Dependencies](#installation)
* [Usage](#Usage)
  * [Pre-processing](#pre-processing)
  * [Training](#training)
  * [Prediction](#prediction)
  * [Evaluation](#evaluation)
  * [Single Song Transcription](#single-song-transcription)
  * [Extra](#extra)
    * [Print Piano Roll](#print-piano-roll)

## Overview

One of the main topic in AMT is to transcribe a given raw audio file into symbolic form, that is transformation from wav to midi.  And our work is the middle stage of this final goal, which we first transcribe the audio into what we called "frame level" domain. This means we split time into frames, and the length of each frame is 88, corresponding to the piano rolls. And then make prediction of which key is in activation. 

Here is an example output:

![maps](./figures/MAPS_1.png)

The top row is the predicted piano roll, and the bottom row is the original label. Colors blue, green, and red represent true-positive, false-positive, and false-negative respectively.

We used semantic segmentation model for transcription, which is also widely used in the field of image processing.  This model is originally improved from DeepLabV3+, and further combined with U-net architecture and focal loss, Illustrated as below: 

![model](./figures/ModelArch.png)

## Installation
To install the requirement, enter the following command:

```
    python3 setup.py install
```

## Usage

#### Pre-processing

1. Download dataset from the official website of MAPS and MusicNet.

2. run the command to pre-process the audios

   for MusicNet, `cd MusicNet/`  and execute this command:

```
   python3 FeatureExtraction.py --MusicNet-path <path/to/downloaded/folder>
```

   for MAPS, `cd MAPS/` and execute this command:

   ```
   python3 Maps_FeatureExtraction.py --MAPS-path <path/to/downloaded/folder>
   ```

3. For more detail usage, run `python3 FeatureExtraction.py --help`

#### Training

There are some cases for training, by defining different input feature type and output cases. 

For input, you can either choose using **HCFP** or **CFP** representation, depending on your settings of pre-processed feature.  

For output, you can choose to train on **MPE mode** or **multi-instrument mode**, if you are using MusicNet for training. If you are using MAPS for training, then you can only train on MPE mode.



  To train the model on MusicNet, run the command:

  ```
  python3 TrainModel.py MusicNet \
                        --dataset-path <path/to/extracted/feature> \ 
                        -o <output/model/name>
  ```


  The default case will train on **MPE** by using **CFP** features. You can train on multi-instrument mode by adding `--multi-instruments` flag, or change to use HCFP feature by adding `--use-harmonic` flag.



  There are also some cases you can specify to accelerate the progress. Specify `--use-ram` to load the whole features into the ram if your ram is big enough (at least 64 GB, suggested >100 GB).

  To validate the execution of the training command, you can also specify less epochs and steps by adding `-e 1 -s 500`. 

  And to continue train on a pre-trained model, add `--input-model <path/to/pre-trained/model>`.

  There are also some callbacks being applied to the training. You can find it around *line 145~150* in *TrainModel.py*.

#### Evaluation

##### *NOTICE: For a whole and complete evaluation process, please check the version 1 code in **v1** folder.*

To predict and evaluate the scores with label, run the command:

```
python3 Evaluation.py  MusicNet \
                       --model_path <path/to/trained/model> \
                       --save-pred <path/to/store/predictions> \
```

Currently, this script doesn't exactly have any evaluation functions. The only use is to make predictions and store them. You should implement the evaluation by yourself, or check out the original code inside **v1** folder.

#### Single Song Transcription

To transcribe on a single song, run the command:

```
python3 SingleSongTest.py --input-audio <input/audio> 
                          --model-path <path/to/pre-trained/model>
```

There will be an output file under the same path named *pred.hdf*, which contains the prediction of the given audio. 

To get the predicted midi, add `--to-midi <path/to/store/midi>` flag. The midi will be stored at the given path.

#### Extra

###### Print Piano Roll

To print out the predictions as images, like above shown, run the command:

```
python3 PrintPianoRoll.py -p <path/to/prediction>
```

The path to the folder should containing *pred.hdf* and *label.hdf*. For each figure, there will at most 4 rows, and two as a group, presenting prediction row and label row to the same piece. If there is no *label.hdf* file, the label row would be the same as prediction row.

The default setting will print original output values, without thresholding.  If you want to print a thresholded figure, add `--quantize` flag. 

To specify output path and figure name, add `-o <path/to/output> -f <figure_name>`.

Notice that if turn on both `--quantize` and `--spec-instrument` to print out some specific instrument channels, you will also need to specify the flag:`--threshold <[list of thresholds]>`, with the same length of specified instruments.

