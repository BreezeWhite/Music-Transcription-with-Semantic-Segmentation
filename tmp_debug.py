
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from  librosa import hz_to_midi

T_UNIT = 0.02
FS = int(1/T_UNIT)

def onsets_info_to_roll(intervals, pitches):
    last_sec = max([t[1] for t in intervals])
    total_frame = last_sec*FS + 200
    roll = np.zeros((total_frame.astype('int'), 128))

    for idx, it in enumerate(intervals):
        midi_num = int(round(hz_to_midi(pitches[idx])))
        start_frm, end_frm = int(it[0]*FS), int(it[1]*FS)
        roll[start_frm:end_frm, midi_num] = 1

    return roll

def plot_onsets_info(ref_intervals, ref_pitches, est_intervals, est_pitches):
    print("Pred len:", len(est_pitches))
    print("Label len:", len(ref_pitches))
    pred = onsets_info_to_roll(est_intervals, est_pitches)
    label = onsets_info_to_roll(ref_intervals, ref_pitches)

    padding = np.abs(len(pred)-len(label)).astype('int')
    padding = np.zeros((padding, 128))
    if len(pred) < len(label):
        pred = np.concatenate([pred, padding])
    elif len(label) < len(pred):
        label = np.concatenate([label, padding])

    roll = pred*9 + label*2

    plot_range = range(0, min(len(roll), 1001))
    roll = roll[plot_range]
    draw(-roll, "debug1.png")

def draw(roll, save_name="debug.png"):
    cmap = "terrain"
    #cmap = "viridis"
    plt.imshow(roll.transpose(), origin="lower", aspect="auto", cmap=cmap)
    plt.savefig(save_name, dpi=250)

if __name__ == "__main__":
    base = "/media/whitebreeze/maps/data/test_feature2/prediction/Maestro-Attn-W4.2"
    pred_path = base + "_predictions.hdf"
    label_path = base + "_labels.pickle"
