#!/bin/bash

cd ..

AUDIO="TestZone/MAPS_MUS-mz_331_3_ENSTDkCl.wav"
MODEL=model/Maestro-Attn-Note-Smooth
TH=5.5
MIDI_NAME="mz.mid"

python3 SingleSongTest.py \
    -i "$AUDIO"    \
    -m $MODEL      \
    --onset-th $TH \
    --to-midi "$MIDI_NAME"

