


MusicNet_Instruments = ["Piano", "Harpsichord", "Violin", "Viola", "Cello", "Contrabass", 
                        "Horn", "Oboe", "Bassoon", "Clarinet", "Flute"]
MusicNetMIDIMapping = {
    "Piano": 1,
    "Harpsichord": 7,
    "Violin": 41,
    "Viola": 42,
    "Cello": 43,
    "Contrabass": 44,
    "Horn": 61,
    "Oboe": 69,
    "Bassoon": 71,
    "Clarinet": 72,
    "Flute": 74
}
HarmonicNum = 5

def get_MusicNet_label_num_mapping(offset=1, spec_inst=None):
    ori_num = [1, 7, 41, 42, 43, 44, 61, 69, 71, 72, 74]
    
    mapping = {}

    if spec_inst is None:
        return mapping
    if "All" in spec_inst:
        spec_inst = MusicNet_Instruments

    for idx, name in enumerate(spec_inst):
        ii = MusicNet_Instruments.index(name)
        n = ori_num[ii]
        mapping[n] = idx + offset
    
    return mapping

def get_instruments_num(insts):
    if insts is None:
        instruments = 1
    elif "All" in insts:
        instruments = 11 # For MusicNet
    else:
        instruments = len(insts)
    
    return instruments


class BaseDatasetInfo:
    base_path=""
    label_ext=""
    train_wavs=[]
    test_wavs=[]
    train_labels=[]
    test_labels=[]

    def __init__(self, base_path):
        self.base_path = base_path


class MapsDatasetInfo(BaseDatasetInfo):
    base_path="/media/data/maps"
    label_ext=".txt"
    train_wavs=[
        "MAPS_AkPnBcht_2/AkPnBcht/MUS",
        "MAPS_AkPnBsdf_2/AkPnBsdf/MUS",
        "MAPS_AkPnStgb_2/AkPnStgb/MUS",
        "MAPS_AkPnCGdD_2/AkPnCGdD/MUS",
        "MAPS_SptkBGCl_2/SptKBGCl/MUS",
        "MAPS_StbgTGd2_2/StbgTGd2/MUS"
    ]
    test_wavs= [
        "MAPS_ENSTDkAm_2/ENSTDkAm/MUS",
        "MAPS_ENSTDkCl_2/ENSTDkCl/MUS"
    ]
    train_labels=train_wavs
    test_labels=test_wavs


class MusicNetDatasetInfo(BaseDatasetInfo):
    base_path="/media/data/MusicNet"
    label_ext=".csv"
    train_wavs=["train_data"]
    test_wavs=["test_data"]
    train_labels=["train_labels"]
    test_labels=["test_labels"]


class MaestroDatasetInfo(BaseDatasetInfo):
    base_path="/media/whitebreeze/data/maestro-v1.0.0"
    label_ext=".midi"
    train_wavs=["2004", "2006", "2008", "2009", "2011", "2013", "2014", "2015"]
    test_wavs=["2017"]
    train_labels=train_wavs
    test_labels=test_wavs


class SuDatasetInfo(BaseDatasetInfo):
    base_path="/data/ade/SuDataset"
    label_ext="_full.mid"
    train_wavs=[]
    test_wavs=[
        "1 Tchaikovsky", "2 schumann", "3 beethoven", "5 Mozart",
        "PQ01_Dvorak", "PQ02_Elgar", "PQ03_Farranc", "PQ04_Frank", "PQ05_Hummel",
        "PQ06_Schostakovich", "PQ07_Schubert", "PQ08_Schubert",
        "SQ01_Beethoven", "SQ02_Janacek", "SQ03_Schubert", "SQ04_Janacek",
        "SQ04_Ravel", "SQ05_Mozart", "SQ07_Haydn", "SQ08_Dvorak", "SQ09_Ravel",
        "SY06_Mahler",
        "VS01_Schumann", "VS02_Brahms", "VS03_Debussy", "VS04_Franck", "VS05_Mozart",
        "WQ01_Nielsen", "WQ02_Schoenberg", "WQ03_Cambini", "WQ04_Danzi"
    ]
    train_labels=train_wavs
    test_labels=test_wavs
    

class URMPDatasetInfo(BaseDatasetInfo):
    base_path="/data/ade/URMP"
    label_ext=".mid"
    train_wavs=[]
    test_wavs=["wavs"]
    train_labels=train_wavs
    test_labels=["labels"]


class Bach10DatasetInfo(BaseDatasetInfo):
    base_path="/data/ade/bach10"
    label_ext=".mid"
    train_wavs=[]
    test_wavs=["wavs"]
    train_labels=train_wavs
    test_labels=["labels"]

