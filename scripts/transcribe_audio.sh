#!/bin/bash

cd ..
AUDIO="TestZone/Angel Beats! piano medley - Animenz Live 2017 in Shenzhen.wav"
AUDIO="TestZone/Swordland (Main Theme)- Sword Art Online OST [piano].wav"
#AUDIO="TestZone/Angel Beats - Brave Song.wav"
#AUDIO="TestZone/My Hero Academia - Boku no Hero Academia OST.wav"
MODEL=model/ICASSP-2019-Maestro-Note

python3 SingleSongTest.py \
    -i "$AUDIO"  \
    -m $MODEL    \
    --to-midi "Swordland_icassp.mid"
