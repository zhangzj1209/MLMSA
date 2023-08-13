class Config(object):
    def __init__(self):
        # Path
        self.root = '/data/MLMSA/K-shape'

        # Other parameters
        self.original_signal_length = 1500   # data length 6s * 250Hz
        self.sample = 250       # Hz
        self.resample = 125     # Hz
        self.new_signal_length = int(self.original_signal_length / (self.sample / self.resample))

        self.lowpass = 25       # lowpass 25Hz
