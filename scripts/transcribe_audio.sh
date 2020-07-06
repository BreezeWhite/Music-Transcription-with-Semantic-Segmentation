#!/bin/bash

cd ..
AUDIO="TestZone/Angel Beats! piano medley - Animenz Live 2017 in Shenzhen.wav"
AUDIO="TestZone/Swordland (Main Theme)- Sword Art Online OST [piano].wav"
#AUDIO="TestZone/Angel Beats - Brave Song.wav"
#AUDIO="TestZone/My Hero Academia - Boku no Hero Academia OST.wav"
#AUDIO="TestZone/Unravel - Tokyo Ghoul OP [Piano].wav"
#AUDIO="TestZone/Level 5 judgelight - To Aru Kagaku no Railgun OP 2 [Piano].wav"
#AUDIO="TestZone/aLIEz - Aldnoah.zero ED 2 [Piano].wav"
#AUDIO="TestZone/Kokoro no Senritsu - Tari Tari Insert song [Piano].wav"
#AUDIO="TestZone/MAPS_MUS-mz_331_3_ENSTDkCl.wav"
#AUDIO="TestZone/Only my Railgun - To aru kagaku no railgun OP1 [full version] [Piano].wav"
#AUDIO="TestZone/Nier Automata OST [Piano].wav"
#AUDIO="TestZone/Cfasman - Concert Fantasy on a theme by Matvey Blanter.wav"
#AUDIO="TestZone/Mahou Shoujo Madoka Magica - Soundtrack Medley [Piano].wav"
#AUDIO="TestZone/Brave Shine - Fatestay night UBW OP2 [Piano].wav"
#AUDIO="TestZone/Zen Zen Zense - Kimi no Na wa OST [piano].wav"
#AUDIO="TestZone/Kiseijuu OP - Let Me Hear.wav"
#AUDIO="TestZone/Girl Inside - Honkai Impact 3 OP [Piano].wav"
#AUDIO="TestZone/Owari no Sekai Kara - Yanagi Nagi [Piano].wav"
#AUDIO="TestZone/All of Me (Jon Schmidt original tune) - The Piano Guys.wav"
#AUDIO="TestZone/Waterfall - The Piano Guys.wav"
#AUDIO="TestZone/Cant Help Falling in Love - The Piano Guys.wav"
#AUDIO="TestZone/Departures - Guilty Crown ED1 [Piano].wav"
#AUDIO="TestZone/Lang Lang - Lisztss Hungarian Rhapsody No2.wav"
#AUDIO="TestZone/Snow Halation - Love Live! OST [Piano].wav"
#AUDIO="TestZone/This Game - No Game No Life OP [Piano].wav"
AUDIO="TestZone/Kimi no Shiranai Monogatari - Bakemonogatari ED [Piano].wav"

#MODEL=model/Maestro-Attn-Note-Smooth-SingleGPU # Threshold: 5.5
#MODEL=model/Maestro-Maps-Attn-Note-Smooth # Threshold: 7
#MODEL=model/ICASSP-2019-Maestro-Note # Threshold: 5.5
#MODEL=model/Maestro-Smooth-Ultimate-Attn-V1 # Threshold: 5.5
MODEL=model/Maestro-Smooth-Ultimate-Attn-SingleGPU
#MODEL=model/Attn-Focal-LS-Strange # Threshold: 5.5
#MODEL=model/Maestro-Smooth-Ultimate # Threshold: 5.5
#MODEL=model/Maestro-Attn-V4.2.1 # Threshold 5
#MODEL=model/Maestro-Smooth-G0.2 # Threshold 5.5
#MODEL=model/Dilated-Conv-Maestro-Note-Smooth # Threshold 5.2
MODEL=model-paper/full_model_all_feature

TH=4
MIDI_NAME="kimi_full.mid"

python3 SingleSongTest.py \
    -i "$AUDIO"    \
    -m $MODEL      \
    --onset-th $TH \
    --to-midi "$MIDI_NAME"
