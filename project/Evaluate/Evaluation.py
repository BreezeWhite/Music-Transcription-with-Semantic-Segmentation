import os
import glob
import math
import h5py
import pickle
import logging
import librosa
import mir_eval
import numpy as np

from project.configuration import MusicNetMIDIMapping, MusicNet_Instruments
from project.utils import load_model, model_info
from project.Evaluate.predict import predict
from project.Evaluate.eval_utils import * 
from project.postprocess import PostProcess, down_sample

from tmp_debug import plot_onsets_info, draw


class EvalEngine:
    @classmethod
    def eval(cls, 
             generator, 
             eval_func, 
             mode="note", 
             onset_th=7, 
             dura_th=1, 
             frm_th=1, 
             t_unit=0.02):
        lowest_pitch = librosa.note_to_midi("A0")
        prec, rec, fs = [], [], []
        for idx, (pred, label, key) in enumerate(generator(), 1):
            print("{}. {}".format(idx, key))
            
            # Create some variable and validate the threshold
            ch_per_inst = 0
            use_list_th = False
            on_th = [onset_th for i in range(pred.shape[-1])]
            du_th = [dura_th for i in range(pred.shape[-1])]
            fr_th = [frm_th for i in range(pred.shape[-1])]
            if mode=="note":
                ch_per_inst = 2
                if isinstance(onset_th, list) or isinstance(dura_th, list):
                    assert(len(onset_th)==len(dura_th)==pred.shape[-1]-1)
                    on_th = onset_th
                    du_th = dura_th
            elif mode=="frame":
                ch_per_inst = 1
                pred = np.dstack([pred[:,:,0], norm(pred[:,:,1:])])
                if isinstance(frm_th, list):
                    assert(len(frm_th)==pred.shape[-1]-1)
                    use_list_th = True
                    fr_th = frm_th
            
            iters = (pred.shape[-1]-1) // ch_per_inst
            ch_p, ch_r, ch_f = 0, 0, 0
            for i in range(iters):
                ch_list = [0] + [i*ch_per_inst+k for k in range(1, ch_per_inst+1)]
                sub_pred = pred[:,:,ch_list]
                midi = PostProcess(sub_pred, mode=mode, onset_th=on_th[i], dura_th=du_th[i], frm_th=fr_th[i])
                inst = MusicNet_Instruments[i]
                inst_num = MusicNetMIDIMapping[inst]
                
                ####### Comment me
                #midi.write("{}_{}.mid".format(key, inst)) if mode=="note" else midi.write("{}_{}_frame.mid".format(key, inst))
                #draw(sub_pred[:,:,1], save_name="{}_{}.png".format(key, inst))
                #######

                p, r, f = eval_func(midi, label, inst_num=inst_num)
                ch_p += p
                ch_r += r
                ch_f += f
                print("\t{} Prec: {:.4f}, Rec: {:.4f}, F: {:.4f}".format(MusicNet_Instruments[i], p, r, f))

            ch_p /= iters
            ch_r /= iters
            ch_f /= iters
            print("Final score. Prec: {:.4f}, Rec: {:.4f}, F: {:.4f}".format(ch_p, ch_r, ch_f))
            prec.append(ch_p)
            rec.append(ch_r)
            fs.append(ch_f)
        return prec, rec, fs

    @classmethod
    def evaluate_frame(cls, pred, label, inst_num=1, t_unit=0.02):
        #inst_num = MusicNetMIDIMapping["Piano"]

        # The input pred should be thresholded
        est_time, est_hz = gen_frame_info_from_midi(pred, t_unit=t_unit) 
        ref_time, ref_hz = gen_frame_info_from_label(label, inst_num=inst_num, t_unit=t_unit)
        out = mir_eval.multipitch.metrics(ref_time, ref_hz, est_time, est_hz)

        precision, recall, accuracy = out[0:3]
        prec_chroma, rec_chroma, acc_chroma = out[7:10]
        fscore = 2*precision*recall/(precision+recall+1e-8)
        return precision, recall, fscore

    @classmethod
    def evaluate_onsets(cls, pred, label, inst_num=1, t_unit=0.02):
        #inst_num = MusicNetMIDIMapping["Piano"]

        # The input pred should be thresholded
        est_interval, est_hz = gen_onsets_info_from_midi(pred, t_unit=t_unit)
        ref_interval, ref_hz = gen_onsets_info_from_label(label, inst_num=inst_num, t_unit=t_unit)
        #plot_onsets_info(ref_interval, ref_hz, est_interval, est_hz)

        try:
            out = mir_eval.transcription.precision_recall_f1_overlap(
                ref_interval, ref_hz, 
                est_interval, est_hz, 
                offset_ratio=None,
                onset_tolerance=0.06
            )
        except ValueError as expt:
            print(expt)
            out = [0, 0, 0, 0]

        precision, recall, fscore, avg_overlap_ratio = out
        return precision, recall, fscore

    @classmethod
    def evaluate_dataset(cls, 
                         mode,
                         feature_path=None,
                         model_path=None,
                         pred_save_path=None,
                         pred_path=None,
                         label_path=None,
                         onset_th=7, 
                         dura_th=1, 
                         frm_th=3,
                         t_unit=0.02):
        """
        Parameters:
            mode: Either one of "note" or "frame".
            feature_path: Path to the generated feature(*.hdf) and label(*.pickle) files.
            pred_path: Path to the prediction hdf file
            pred_save_path: Path to save the prediction, optional.
            label_path: Path to the label hdf file generated after executeing predict_dataset().
            onset_th: onset threshold (note mode)
            dura_th: duration channel threshold (note mode)
            frm_th: frame threshold (frame mode)
            t_unit: Time unit of each frame in second.
            
        You should either provide one of (feature_path, model_path) pair or (pred_path, label_path) pair.
        """
        if mode not in ["note", "frame"]:
            err_info = "Please specify the mode with one of the value 'note' or 'frame'."
            raise ValueError(err_info)
        else:
            eval_func = cls.evaluate_onsets if mode=="note" else cls.evaluate_frame

        if feature_path is not None and model_path is not None:
            print("Predicting on the dataset: %s", feature_path)
            generator = lambda: cls.predict_dataset(feature_path, model_path, output_save_path=pred_save_path)
        elif pred_path is not None and label_path is not None:
            cont = []
            pred_f = h5py.File(pred_path, "r")
            print("Loading labels")
            label_f = pickle.load(open(label_path, "rb"))
            print("Loading predictions")
            for key, pred in pred_f.items():
                #### Comment me
                #if key != "1759":
                #    continue
                ####
                pred = pred[:]
                ll = label_f[key]
                cont.append([pred, ll, key])

            pred_f.close()
            generator = lambda: cont
        else:
            raise ValueError("Unknown parameter combination")

        lprec, lrec, lfs = cls.eval(generator, eval_func, mode=mode, onset_th=onset_th, dura_th=dura_th, frm_th=frm_th)
        length = len(lprec)
        avg_prec = sum(lprec)/length
        avg_rec = sum(lrec)/length
        avg_fs = sum(lfs)/length
        print("Precision: {:.4f}, Recall: {:.4f}, F-score: {:.4f}".format(avg_prec, avg_rec, avg_fs))
        return avg_prec, avg_rec, avg_fs

    @classmethod
    def predict_dataset(
        cls,
        feature_path,
        model_path,
        output_save_path=None
    ):
        """
        This is a generator function.
        Parameters:
            feature_path: Path to the generated feature(*.hdf) and label(*.pickle) files. 
            model_path: Path to the pre-trained model.
            output_save_path: Path to save predictions(include labels) if not none.
        """
        write_pred = lambda ff, idx: 0
        if output_save_path is not None:
            if not os.path.exists(output_save_path):
                os.makedirs(output_save_path)
            pred_out_name = os.path.basename(model_path)+"_predictions.hdf"
            pred_out_path = os.path.join(output_save_path, pred_out_name)
            p_out = h5py.File(pred_out_path, "w")
            write_pred = lambda ff, idx: p_out.create_dataset(str(idx), data=ff, compression="gzip", compression_opts=5)

        labels = {}
        hdfs = glob.glob(os.path.join(feature_path, "*.hdf"))
        if len(hdfs) == 0:
            raise ValueError("No feature files found at path {}".format(feature_path))

        for (pred, ll, key) in cls.predict_hdf(hdfs, model_path):
            write_pred(pred, key)
            labels[key] = ll
            yield pred, ll, key

        p_out.close() if p_out is not None else None
        label_out_name = os.path.basename(model_path)+"_labels.pickle"
        pickle.dump(labels, open(os.path.join(output_save_path, label_out_name), "wb"), pickle.HIGHEST_PROTOCOL)

    @classmethod
    def predict_hdf(
        cls,
        hdf_paths,
        model_path,
        pred_batch_size=4
    ):
        """
        This is a generator function.
        Assert there exist corresponding label files with extension .pickle under the same
        directory of given hdf_paths.
        """
        
        if not isinstance(hdf_paths, list):
            hdf_paths = [hdf_paths]
        
        model = load_model(model_path)
        feature_type, channels, out_classes, timesteps = model_info(model_path)
        
        for hdf_path in hdf_paths:
            with h5py.File(hdf_path, "r") as feat:
                label_path = hdf_path.replace(".hdf", ".pickle")
                label = pickle.load(open(label_path, "rb"))
                for key, ff in feat.items():
                    ff = create_batches(ff[:,:,channels], b_size=pred_batch_size, timesteps=timesteps) 
                    ll = label[key]
                    pred = predict(ff, model)
                    
                    yield pred, ll, key


