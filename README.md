# Music Transcription with Semantic Model

## About

This is a AMT (Automatic  Music Transcription) project, combined with state-of-the-art image semantic segmentation neural network model. 

The dataset used is MAPS and MusicNet, which the first one is a solo-piano performance collection, and the second is a multi-instrument performance collection.  On both dataset, we achieved the state-of-the-art results on MPE (Multi-Pitch Estimation) case. 

This work was done based on our prior work of [repo1](https://github.com/BreezeWhite/CFP_NeuralNetwork), [repo2](https://github.com/s603122001/Vocal-Melody-Extraction)

The original paper is under reviewing. Coming soon....

## Overview

One of the main topic in AMT is to transcribe a given raw audio file into symbolic form, that is transformation from wav to midi.  And our work is the middle stage of this final goal, which we first transcribe the audio into what we called "frame level" domain. This means we split time into frames, and the length of each frame is 88, corresponding to the piano rolls. And then make prediction of which key is in activation. 

Here is an example output:

![maps](./figures/MAPS_1.png)

The top row is the predicted piano roll, and the bottom row is the original label. Colors blue, green, and red represent true-positive, false-positive, and false-negative respectively.

We used semantic segmentation model for transcription, which is also widely used in the field of image processing.  This model is originally improved from DeepLabV3+, and further combined with U-net architecture and focal loss, Illustrated as below: 

![model](./figures/ModelArch.png)



## Usage

- Pre-processing

  Download dataset from MAPS and MusicNet.

- Training

  (some command here...)

- Testing

  (some command here...)

- Extra

  Print out the prediction result.