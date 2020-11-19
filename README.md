
# Music Transcription with Semantic Model

### Notice - A new project has been launched, which also contains this work. Please visit [omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart).

This is a Automatic  Music Transcription (AMT) project, aim to deal with **Multi-pitch Estimation** (MPE) problem, which has been a long-lasting and still a challenging problem. For the transcription, we leverage the state-of-the-art image semantic segmentation neural network and attention mechanism for transcribing piano solo, and also multi-instrument performances. 

The dataset used is MAPS and MusicNet, which the first one is a solo-piano performance collection, and the second is a multi-instrument performance collection.  On both dataset, we achieved the state-of-the-art results on MPE (Multi-Pitch Estimation) case frame-wisely, which on **MAPS** we achieved **F-score 86.73%**, and on **MusicNet** we achieved **F-score 73.70%**.

This work was done based on our prior work of [repo1](https://github.com/BreezeWhite/CFP_NeuralNetwork), [repo2](https://github.com/s603122001/Vocal-Melody-Extraction). For more about our works, please meet our [website](https://sites.google.com/view/mctl/home).

For whom would interested in more technical details, the original paper is [here](https://ieeexplore.ieee.org/abstract/document/8682605).

## Quick Start

The most straight forward way to enjoy our project is to use our [colab](http://bit.ly/transcribe-colab). Just press the start button cell by cell, and you cant get the final output midi file of the given piano clip.

A more technical way is to download this repository by executing `git clone https://github.com/BreezeWhite/Music-Transcription-with-Semantic-Segmentation.git`, and then enter *scripts* folder, modify `transcribe_audio.sh`, then run the script.


## Table of Contents

* [Quick Start](#quick-start)

* [Overview](#overview)

* [Install Dependencies](#installation)

* [Usage](#Usage)
  * [Pre-processing](#pre-processing)

  * [Training](#training)

  * [Evaluation](#evaluation)

  * [Single Song Transcription](#single-song-transcription)

    

## Overview

One of the main topic in AMT is to transcribe a given raw audio file into symbolic form, that is transformation from wav to midi.  And our work is the middle stage of this final goal, which we first transcribe the audio into what we called "frame level" domain. This means we split time into frames, and the length of each frame is 88, corresponding to the piano rolls. And then make prediction of which key is in activation. 

Here is an example output:

![maps](./figures/MAPS_1.png)

The top row is the predicted piano roll, and the bottom row is the original label. Colors blue, green, and red represent true-positive, false-positive, and false-negative respectively.

We used semantic segmentation model for transcription, which is also widely used in the field of image processing.  This model is originally improved from DeepLabV3+, and further combined with U-net architecture and focal loss, Illustrated as below: 

![model](./figures/ModelArch.png)

## Installation
To install the requirements, enter the following command:

```
pip install -r requirements.txt
```
Download weights of the check points:
```
git lfs fetch
```

## Usage

For a quick example usage, you can enter *scripts* folder and check the code in the script to see how to use the python code.

#### Pre-processing

1. Download dataset from the official website of MAPS and MusicNet.

2. ```cd scripts```

3. Modify the content of *generate_feature.sh*

4. Run the *generate_feature.sh* script

#### Training

There are some cases for training, by defining different input feature type and output cases. For quick start, please refer to *scripts/train_model.sh* 

For input, you can either choose using **HCFP** or **CFP** representation, depending on your settings of pre-processed feature.  

For output, you can choose to train on **MPE mode** or **multi-instrument mode**, if you are using MusicNet for training. If you are using MAPS for training, then you can only train on MPE mode.



  To train the model on MusicNet, run the command:

  ```
  python3 TrainModel.py MusicNet \
      <output/model/name>
      --dataset-path <path/to/extracted/feature> \ 
  ```


  The default case will train on **MPE** by using **CFP** features. You can train on multi-instrument mode by adding `--multi-instruments` flag, or change to use HCFP feature by adding `--use-harmonic` flag.



  There are also some cases you can specify to accelerate the progress. Specify `--use-ram` to load the whole features into the ram if your ram is big enough (at least 64 GB, suggested >100 GB).

  To validate the execution of the training command, you can also specify less epochs and steps by adding `-e 1 -s 500`. 

  And to continue train on a pre-trained model, add `--input-model <path/to/pre-trained/model>`.

#### Evaluation

##### *NOTICE: For a whole and complete evaluation process, please check the version 1 code in **v1** folder.*

To predict and evaluate the scores with label, run the command:

```
python3 Evaluation.py frame \
    --feature-path <path/to/generated/feature> \
    --model-path <path/to/trained/model> \
    --pred-save-path <path/to/store/predictions> \
```

You can check out *scripts/evaluate_with_pred.sh* and *scripts/pred_and_evaluate.sh* for example use.

#### Single Song Transcription

To transcribe on a single song, run the command:

```
python3 SingleSongTest.py \
    --input-audio <input/audio> 
    --model-path <path/to/pre-trained/model>
```

There will be an output file under the same path named *pred.hdf*, which contains the prediction of the given audio. 

To get the predicted midi, add `--to-midi <path/to/save/midi>` flag. The midi will be stored at the given path.

There is also an example script in *scripts* folder called *transcribe_audio.sh*


