
import mir_eval

from project.Evaluate.eval_utils import gen_frame_info, gen_onsets_info


def evaluate_onsets(pred, label, t_unit=0.02):
    # The input pred should be thresholded

    est_interval, est_hz = gen_onsets_info(pred, t_unit=t_unit)
    ref_interval, ref_hz = gen_onsets_info(label, t_unit=t_unit)
    out = mir_eval.transcription.precision_recall_f1_overlap(ref_interval, ref_hz, 
                                                             est_interval, est_hz, 
                                                             offset_ratio=None)
    precision, recall, fscore, avg_overlap_ratio = out

    return precision, recall, fscore

def evaluate_frame(pred, label, t_unit=0.02):
    # The input pred should be thresholded

    est_time, est_hz = gen_frame_info(pred, t_unit) 
    ref_time, ref_hz = gen_frame_info(label, t_unit)
    out = mir_eval.multipitch.metrics(ref_time, ref_hz, est_time, est_hz)

    precision, recall, accuracy = out[0:3]
    prec_chroma, rec_chroma, acc_chroma = out[7:10]

    fscore = 2*precision*recall/(precision+recall+1e-8)

    return precision, recall, fscore

def evaluation(preds, labels, onsets=False):
    # The input preds should be thresholded

    eval_func = evaluate_onsets if onsets else evaluate_frame
    
    precision, recall, f1 = 0, 0, 0
    l_prec, l_rec, l_f = [], [], []
    
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]

        prec, rec, f = eval_func(pred, label)
        
        precision += prec
        recall += rec
        f1 += f

        l_prec.append(prec)
        l_rec.append(rec)
        l_f.append(f)

    precision /= len(preds)
    recall /= len(preds)
    f1 /= len(preds)

    return precision, recall, f1, l_prec, l_rec, l_f
