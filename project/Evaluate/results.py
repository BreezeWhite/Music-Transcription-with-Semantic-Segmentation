import os
import csv
from collections import namedtuple
import numpy as np


Result = namedtuple("Result", ["inst_name", "precision", "recall", "fscore", "inst_acc", "overlap"])
    
class EvalResults:
    def __init__(self):
        self.results = []
        self._offsets = {}

    def add_result(self, inst, prec, rec, f, inst_acc, avg_overlap=0):
        result = Result(inst_name=inst, precision=prec, recall=rec, fscore=f, inst_acc=inst_acc, overlap=avg_overlap)
        self.results.append(result)

    def add_final_result(self, key, prec, rec, f, inst_acc, avg_overlap=0):
        result = Result(inst_name="Avg", precision=prec, recall=rec, fscore=f, inst_acc=inst_acc, overlap=avg_overlap)
        self.results.append(result)
        self._offsets[key] = len(self.results)

    def get_each_avg(self):
        each = {}
        each_count = {}
        for result in self.results:
            if result.inst_name not in each:
                each[result.inst_name] = {"precision": 0, "recall": 0, "fscore": 0, "inst_acc": 0, "overlap": 0}
                each_count[result.inst_name] = 0
            each[result.inst_name]["precision"] += result.precision
            each[result.inst_name]["recall"] += result.recall
            each[result.inst_name]["fscore"] += result.fscore
            each[result.inst_name]["inst_acc"] += result.inst_acc
            each[result.inst_name]["overlap"] += result.overlap
            each_count[result.inst_name] += 1
        for key in each.keys():
            cnt = each_count[key]
            each[key]["precision"] /= cnt
            each[key]["recall"] /= cnt
            each[key]["fscore"] /= cnt
            each[key]["inst_acc"] /= cnt
            each[key]["overlap"] /= cnt
        return each
    
    def write_results(self, out_path):
        if not out_path.endswith(".csv"):
            out_path = os.path.join(out_path, "results.csv")
        dir_name = os.path.dirname(out_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(out_path, "w") as out_file:
            writer = csv.DictWriter(
                out_file, 
                delimiter=',',
                fieldnames=["Test set", "inst_name", "precision", "recall", "fscore", "inst_acc", "overlap"]
            )
            writer.writeheader()

            # Add total average first
            each_avg = self.get_each_avg()
            for k, v in each_avg.items():
                writer.writerow({"Test set": "Total average", "inst_name": k, **v})
            
            # Add each result
            offset = 0
            for k, v in self._offsets.items():
                rr = range(offset, v)
                for i in range(offset, v):
                    row = {"Test set": k, **self.results[i]._asdict()}
                    writer.writerow(row)
                offset = v

    def get_avg(self):
        overall_avg = self.get_each_avg()
        avg = overall_avg["Avg"]
        return avg["precision"], avg["recall"], avg["fscore"], avg["inst_acc"], avg["overlap"]

    def __str__(self):
        score_str = "Prec: {:.4f}, Rec: {:.4f} F-score: {:.4f} Inst Acc: {:.4f} Overlap: {:.4f}".format(
            *self.get_avg()
        )

        return score_str

