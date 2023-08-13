class Config(object):
    def __init__(self):
        # Path
        self.root = '/data/MLMSA/xcorr'

        # Other parameters
        self.original_signal_length = 1500   # data length 6s * 250Hz
        self.sample = 250       # Hz
        self.window = 6         # sec
        self.step = 4           # sec
