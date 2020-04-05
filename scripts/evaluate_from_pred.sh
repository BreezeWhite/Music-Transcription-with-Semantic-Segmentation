#!/bin/bash

### Maestro ###
MODEL_NAME=Maestro-Attn-Note-Smooth
PRED_FOLDER=maestro_attn_note_smooth
TH=6

#MODEL_NAME=Maestro-Smooth-G0.2
#PRED_FOLDER=maestro_smooth_g0.2
#TH=5

#MODEL_NAME=Maestro-Smooth-Ultimate-Attn-V1
#PRED_FOLDER=maestro_smooth_ultimate_attn_v1
#TH=4.5


#MODEL_NAME=Maestro-Smooth-Ultimate
#PRED_FOLDER=maestro_smooth_ultimate
#TH=5

#MODEL_NAME=Dilated-Conv-Maestro-Note-Smooth
#PRED_FOLDER=dilated_conv_maestro_note_smooth
#TH=5

#MODEL_NAME=Maestro-Maps-V4.2.1
#PRED_FOLDER=maestro_maps_v4.2.1
#TH=7.5

#MODEL_NAME=Maestro-Attn-V4.2.1
#PRED_FOLDER=maestro_attn_v4.2.1
#TH=5

### MusicNet ###
MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
PRED_FOLDER=musicnet_smooth_v1.0.2
TH=8

#MODEL_NAME=MusicNet-HCFP
#PRED_FOLDER=musicnet_hcfp
#TH=7

### Additional Dataset ###
MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
PRED_FOLDER=su_smooth_v1.0.2
TH=8

#MODEL_NAME=Dilated-Conv-MusicNet-Note-Smooth
#PRED_FOLDER=su_icassp_2019_LS
#TH=8

#MODEL_NAME=ICASSP-2019-MusicNet-Note
#PRED_FOLDER=su_icassp_2019
#TH=7

#MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
#PRED_FOLDER=maps_on_musicnet_smooth_v1.0.2
#TH=8

MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
PRED_FOLDER=urmp_22_smooth_v1.0.2
TH=8

#MODEL_NAME=Dilated-Conv-MusicNet-Note-Smooth
#PRED_FOLDER=urmp_22_icassp_2019_LS
#TH=8

#MODEL_NAME=ICASSP-2019-MusicNet-Note
#PRED_FOLDER=urmp_22_icassp_2019
#TH=7


### Old Convolution-based Model ###
#MODEL_NAME=Dilated-Conv-MusicNet-Note-Smooth
#PRED_FOLDER=dilated_conv_musicnet_note_smooth
#TH=6

#MODEL_NAME=ICASSP-2019-Maestro-Note
#PRED_FOLDER=icassp_2019_maestro_note
#TH=7

#MODEL_NAME=ICASSP-2019-Maestro-Frame
#PRED_FOLDER=icassp_2019_maestro_frame

#MODEL_NAME=ICASSP-2019-MusicNet-Note
#PRED_FOLDER=icassp_2019_musicnet_note
#TH=5

#MODEL_NAME=MusicNet_CFP_MPE
#PRED_FOLDER=su_10_icassp_2019_musicnet_mpe
#TH=0.34

MODE=note
PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

#PRED_PATH="./prediction/${PRED_FOLDER}/pred.hdf"
#LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"


cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH  \
    -l $LABEL_PATH \
    --onset-th $TH
    
echo $PRED_FOLDER
