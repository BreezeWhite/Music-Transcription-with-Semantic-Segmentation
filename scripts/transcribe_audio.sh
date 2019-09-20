#!/bin/bash

cd ..
AUDIO="Angel Beats! piano medley - Animenz Live 2017 in Shenzhen.wav"
MODEL=model/Maestro-Onset-Frame-Attn

python3 SingleSongTest.py \
    -i "$AUDIO"  \
    -m $MODEL \
    --to-midi ./pred.mid
