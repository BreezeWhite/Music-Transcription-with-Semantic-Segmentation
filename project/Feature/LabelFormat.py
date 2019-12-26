
class LabelFmt:
    def __init__(self, start_time, end_time, note, instrument=1,  start_beat=0., end_beat=10., note_value=""):
        """
        Instrument number
            1  Piano
            7  Harpsichord
            41 Violin
            42 Viola
            43 Cello
            44 Contrabass
            61 Horn     
            69 Oboe
            71 Bassoon
            72 Clarinet
            74 Flute
        """
        self.start_time = start_time # in second, float
        self.end_time = end_time # in second, float
        self.instrument = instrument # piano, int
        self.note = note # midi number, int
        self.start_beat = start_beat # float
        self.end_beat = end_beat # float
        self.note_value = note_value 

    def get(self):
        return [self.start_time, self.end_time, self.instrument, self.note, self.start_beat, self.end_beat, self.note_value]
