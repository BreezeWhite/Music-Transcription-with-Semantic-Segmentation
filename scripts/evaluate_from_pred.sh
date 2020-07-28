#!/bin/bash

MODEL_NAME=feature-compare_spec+ceps+gcos
PRED_FOLDER=val_feature-compare_ceps-only
PRED_FOLDER=feature-compare_spec+ceps+gcos
TH=6

# ------------------------------ Maestro -----------------------------------

# Attn-LS
#MODEL_NAME=Maestro-Attn-Note-Smooth
#PRED_FOLDER=maestro_attn_note_smooth
#TH=0.05

# Conv
#MODEL_NAME=ICASSP-2019-Maestro-Note
#PRED_FOLDER=icassp_2019_maestro_note
#TH=0.07

# Conv-LS
#MODEL_NAME=Dilated-Conv-Maestro-Note-Smooth
#PRED_FOLDER=dilated_conv_maestro_note_smooth
#TH=0.07

# Others
#MODEL_NAME=traditional-crossentropy
#PRED_FOLDER=traditional_bce
#TH=0.5

# --------------------------------------------------------------------------
# ----------------------------- MusicNet -----------------------------------
# Attn-LS
MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
PRED_FOLDER=musicnet_smooth_v1.0.2
TH=8

# Conv
#MODEL_NAME=ICASSP-2019-MusicNet-Note
#PRED_FOLDER=icassp_2019_musicnet_note
#PRED_FOLDER=maps_on_musicnet_icassp_2019
#TH=7

#Conv-LS
#MODEL_NAME=Dilated-Conv-MusicNet-Note-Smooth
#PRED_FOLDER=dilated_conv_musicnet_note_smooth
#PRED_FOLDER=maps_on_musicnet_icassp_2019_LS
#TH=8

# --------------------------------------------------------------------------

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
