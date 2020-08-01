#!/bin/bash

cd ..

MODEL=./CheckPoint/MusicNet-CFP-Frame-SingleInst
AUDIO="TestZone/MAPS_MUS-mz_331_3_ENSTDkCl.wav"
MIDI_NAME="mz.mid"

python3 SingleSongTest.py \
    -i "$AUDIO"    \
    -m $MODEL      \
    --to-midi "$MIDI_NAME"
