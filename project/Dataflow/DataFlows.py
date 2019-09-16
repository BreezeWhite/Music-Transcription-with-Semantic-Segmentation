
import os

from project.Dataflow.BaseDataflow import BaseDataflow

class MaestroDataflow(BaseDataflow):
    
    structure = {
        "train":       "feature_train",
        "train_label": "feature_train",
        "val":         "feature_val",
        "val_label":   "feature_val",
        "test":        "feature_test",
        "test_label":  "feature_test"
    }

    def load_data(self, phase, use_ram=False):
        """ 
        Create hdf/pickle instance to the data

        Args:
            phase: "train", "val", or "test"
            use_ram: load data into ram
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

        hdfs = self.parse_files(path, ".hdf")
        feature = self.parse_hdf(hdfs, use_ram=use_ram)
        
        lbs = [a.replace(".hdf", ".pickle") for a in hdfs]
        label = self.parse_pickle(lbs)

        return feature, label
                

class MusicNetDataflow(MaestroDataflow):
    structure = {
        "train":       "features/post_exp/train",
        "train_label": "features/post_exp/train",
        "val":         "features/post_exp/train",
        "val_label":   "features/post_exp/train",
        "test":        "features/post_exp/test",
        "test_label":  "features/post_exp/test"
    }

    def post_init(**kwargs):
        # Split train/val data from training set

        split_p = round(len(self.features) * self.train_val_split)
        if self.phase == "train":
            self.features = self.features[:split_p]
            self.labels = self.labels[:split_p]
        elif self.phase == "val":
            self.features = self.featurs[split_p:]
            self.labels = self.labels[split_p:]

        self.init_index()
        

class MapsDataflow(MusicNetDataflow):
    structure = {
        "train":       "train_feature",
        "train_label": "train_feature",
        "val":         "train_feature",
        "val_label":   "train_feature",
        "test":        "test_feature",
        "test_label": "test_feature"
    }
