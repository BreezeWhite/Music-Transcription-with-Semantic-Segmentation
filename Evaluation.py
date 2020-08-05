import os 
import argparse

from project.Evaluate.Evaluation import EvalEngine

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def create_parser():
    parser = argparse.ArgumentParser("Evaluate one the dataset")
    parser.add_argument("mode", help="To evaluate on note-level or frame-level. One of the 'note' or 'frame'.",
                        type=str, choices=['note', 'frame', 'mpe_note', 'mpe_frame'])
    parser.add_argument("-f", "--feature-path", help="Path to the generated feature(*.hdf) and label(*.pickle) files",
                        type=str, default=None)
    parser.add_argument("-m", "--model-path", help="Path to the pre-trained model",
                        type=str, default=None)
    parser.add_argument("-s", "--pred-save-path", help="Path to save the predictions and labels. Save if provided.",
                        type=str, default=None)
    parser.add_argument("-p", "--pred-path", help="Path to saved prediction file(*_predictions.hdf).",
                        type=str, default=None)
    parser.add_argument("-l", "--label-path", help="Path to the generated label file(*_labels.hdf) while make predictions.",
                        type=str, default=None)
    parser.add_argument("--onset-th", help="Onset threshold (in std)",
                        type=float, default=7)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    results = EvalEngine.evaluate_dataset(
        args.mode, 
        feature_path=args.feature_path,
        model_path=args.model_path,
        pred_save_path=args.pred_save_path,
        pred_path=args.pred_path,
        label_path=args.label_path,
        onset_th=args.onset_th
    )
    
    if not os.path.exists("./result"):
        os.makedirs("./result")
    prefix = args.pred_path.split("/")[-2]
    results.write_results(f"./result/{prefix}_results.csv")


if __name__ == "__main__":
    main()

