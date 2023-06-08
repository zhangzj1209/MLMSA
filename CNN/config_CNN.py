class Config(object):
    def __init__(self):
        # Path
        self.root = 'D:/PyCharm/Code/CNN'

        # Other parameters
        self.sample = 250       # Hz
        self.window = 6         # sec
        self.signal_length = self.sample * self.window   # data length 6s * 250Hz
        self.resample = 125     # Hz
        self.lowpass = 25       # lowpass 25Hz

        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 72
        self.epoch = 20