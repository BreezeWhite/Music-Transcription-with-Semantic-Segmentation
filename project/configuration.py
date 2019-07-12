
#from project.Dataflow.DataFlows import Maestro, MusicNet



MusicNet_Instruments = ["Piano", "Harpsichord", "Violin", "Viola", "Cello", "Contrabass", 
                        "Horn", "Oboe", "Bassoon", "Clarinet", "Flute"]
Harmonic_Num = 6

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

def epsilon():
    return 0.000000001
