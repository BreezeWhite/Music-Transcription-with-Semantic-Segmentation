
import os

from project.Dataflow.BaseDataflow import BaseDataflow

class MaestroDataflow(BaseDataflow):
    
    structure = {
        "train":       "train_feature",
        "train_label": "train_feature",
        "val":         "train_feature",
        "val_label":   "train_feature",
        "test":        "feature_test",
        "test_label":  "feature_test"
    }

    def load_data(self, phase, use_ram=False):
        """ 
        Create hdf/pickle instance to the data

        Args:
            phase: "train", "val", or se_ram: load data into ram
        Returns:
            feature: h5py references to the feature
            label:   specilaized data structure of lables
        """

        if self.structure is None:
            raise NotImplementedError
        
        if phase == "train":
            path = os.path.join(self.dataset_path, self.structure["train"])
            g_path = os.path.join(self.dataset_path, self.structure["train_label"])
        elif phase == "val":
            path = os.path.join(self.dataset_path, self.structure["val"])
            g_path = os.path.join(self.dataset_path, self.structure["val_label"])
        elif phase == "test":
            path = os.path.join(self.dataset_path, self.structure["test"])
            g_path = os.path.join(self.dataset_path, self.structure["test_label"])
        else:
            raise TypeError

        hdfs = self.parse_files(path, ".hdf")#[:1]
        return self.parse_feature_labels(hdfs)

    def post_init(self, **kwargs):
        # Split train/val data from training set
        split_p = round(len(self.features) * self.train_val_split)
        if self.phase == "train":
            self.features = self.features[:split_p]
            self.labels = self.labels[:split_p]
        elif self.phase == "val":
            self.features = self.features[split_p:]
            self.labels = self.labels[split_p:]

        self.init_index()


class MusicNetDataflow(MaestroDataflow):
    structure = {
        "train":       "train_feature/harmonic",
        "train_label": "train_feature/harmonic",
        "val":         "train_feature/harmonic",
        "val_label":   "train_feature/harmonic",
        "test":        "test_feature/harmonic",
        "test_label":  "test_feature/harmonic"
    }

    

class MapsDataflow(MusicNetDataflow):
    structure = {
        "train":       "train_feature",
        "train_label": "train_feature",
        "val":         "train_feature",
        "val_label":   "train_feature",
        "test":        "test_feature",
        "test_label":  "test_feature"
    }
