import time
import torch
import numpy as np
import obspy
from config_CNN import Config
from CNN import predict

start_time = time.time()
config = Config()
root = config.root

sampling = config.sample
window = config.window
step = 4
total_time = 150
total_data_length = sampling * total_time

path = root + '/application_data/section5/'

data_203 = np.empty((3, 1500))
data_203[0, :] = obspy.read(path + '220320.080510.EB000203.EHE.sac')[0].data    # 203为固定台站
data_203[1, :] = obspy.read(path + '220320.080510.EB000203.EHN.sac')[0].data
data_203[2, :] = obspy.read(path + '220320.080510.EB000203.EHZ.sac')[0].data

data_208 = np.empty((3, 6500))
data_208[0, :] = obspy.read(path + '220320.080500.EB000208.EHE.sac')[0].data
data_208[1, :] = obspy.read(path + '220320.080500.EB000208.EHN.sac')[0].data
data_208[2, :] = obspy.read(path + '220320.080500.EB000208.EHZ.sac')[0].data

data_207 = np.empty((3, 11500))
data_207[0, :] = obspy.read(path + '220320.080450.EB000207.EHE.sac')[0].data
data_207[1, :] = obspy.read(path + '220320.080450.EB000207.EHN.sac')[0].data
data_207[2, :] = obspy.read(path + '220320.080450.EB000207.EHZ.sac')[0].data

window_208 = np.lib.stride_tricks.sliding_window_view(data_208, (3, sampling * window)).squeeze()[::sampling * step, :, :]
window_207 = np.lib.stride_tricks.sliding_window_view(data_207, (3, sampling * window)).squeeze()[::sampling * step, :, :]

result = 0
for j in range(len(window_208)):
    for k in range(len(window_207)):
        Test_data = np.concatenate((data_203, window_208[j], window_207[k]), axis=0)
        Test_data = (Test_data - np.min(Test_data, axis=1)[:, np.newaxis]) / (
                np.max(Test_data, axis=1)[:, np.newaxis] - np.min(Test_data, axis=1)[:, np.newaxis])
        Test_data = torch.from_numpy(Test_data)
        Test_data = Test_data.to(torch.float32)
        Test_data = Test_data.unsqueeze_(dim=0)
        result = predict(Test_data)
        if result == 1:
            break
    if result == 1:
        break

print(j, k)
now_time = time.time()
print('Running Time: ', now_time - start_time, 's')