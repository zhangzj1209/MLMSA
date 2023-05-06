# 计算两个序列或多个序列的的时域互相关
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import obspy
from synthetic_tests_lib import crosscorr, compute_shift
from config_xcorr import Config

path = Config().root + '/data'

starttime = time.time()
sampling = Config().sample
window = Config().window
step = Config().step

data_203 = obspy.read(path + '/220320.080510.EB000203.EHZ.sac')[0].data    # 203为固定台站
data_208 = obspy.read(path + '/220320.080500.EB000208.EHZ.sac')[0].data
data_207 = obspy.read(path + '/220320.080450.EB000207.EHZ.sac')[0].data

window_208 = np.lib.stride_tricks.sliding_window_view(data_208, sampling * window)[::sampling * step, :]
window_207 = np.lib.stride_tricks.sliding_window_view(data_207, sampling * window)[::sampling * step, :]

series_203 = pd.Series(data_203.tolist())
RS_203_208 = []
SHIFT_203_208 = []

for i in range(len(window_208)):
    series_208 = pd.Series(window_208[i].tolist())
    lags = np.arange(-750, 750, 1)
    rs_203_208 = np.nan_to_num([crosscorr(series_203, series_208, lag) for lag in lags])
    print('xcorr series 203 and series 208 in windows ', i+1, lags[np.argmax(rs_203_208)], np.max(rs_203_208))

    RS_203_208.append((np.max(rs_203_208)))
    SHIFT_203_208.append(lags[np.argmax(rs_203_208)])

max_corr_203_208 = max(RS_203_208)
print(max_corr_203_208)
print(SHIFT_203_208[RS_203_208.index(max(RS_203_208))] / sampling, 's')

endtime = time.time()
print('Running time: ', endtime - starttime)

fig, ax = plt.subplots(3, 1, figsize=(10, 6))

ax[0].plot(series_203, color='b', label='EB000203')
ax[0].legend()
ax[1].plot(pd.Series(window_208[RS_203_208.index(max(RS_203_208))].tolist()), color='r', label='EB000208')
ax[1].legend()

ax[2].plot(lags, rs_203_208, color='k', label='Time-domain Correlation')
ax[2].legend()
ax[2].axvline(x=lags[np.argmax(rs_203_208)])
plt.savefig('Fig 3-15.jpg', dpi=300, bbox_inches='tight')
plt.show()