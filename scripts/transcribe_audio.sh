#!/bin/bash

cd ..
#AUDIO="TestZone/Angel Beats! piano medley - Animenz Live 2017 in Shenzhen.wav"
AUDIO="TestZone/Swordland (Main Theme)- Sword Art Online OST [piano].wav"
#AUDIO="TestZone/Angel Beats - Brave Song.wav"
#AUDIO="TestZone/My Hero Academia - Boku no Hero Academia OST.wav"
AUDIO="TestZone/Unravel - Tokyo Ghoul OP [Piano].wav"
#AUDIO="TestZone/Level 5 judgelight - To Aru Kagaku no Railgun OP 2 [Piano].wav"
#AUDIO="TestZone/aLIEz - Aldnoah.zero ED 2 [Piano].wav"
#AUDIO="TestZone/Kokoro no Senritsu - Tari Tari Insert song [Piano].wav"
AUDIO="TestZone/MAPS_MUS-mz_331_3_ENSTDkCl.wav"
#AUDIO="TestZone/Only my Railgun - To aru kagaku no railgun OP1 [full version] [Piano].wav"

MODEL=model/Maestro-Attn-Note-Smooth # Threshold: 7.5
#MODEL=model/Maestro-Maps-Attn-Note-Smooth # Threshold: 7
#MODEL=model/ICASSP-2019-Maestro-Note # Threshold: 5.5

TH=7.5
MIDI_NAME="mz_smooth.mid"

python3 SingleSongTest.py \
    -i "$AUDIO"    \
    -m $MODEL      \
    --onset-th $TH \
    --to-midi "$MIDI_NAME"
