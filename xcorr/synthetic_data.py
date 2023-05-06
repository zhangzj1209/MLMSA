# 以滤波平滑后的两序列为例进行互相关说明
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from synthetic_tests_lib import crosscorr, compute_shift
from scipy import signal

# Delta函数
length = 100
amp1, amp2 = 1, 1
x = np.arange(0, length)
to = 10
timeshift = 30
t1 = to+timeshift
series1 = signal.unit_impulse(length, idx=to)  # 单位脉冲
series2 = signal.unit_impulse(length, idx=t1)

# 低通滤波平滑边界(美化图像)
b, a = signal.butter(4, 0.1)  # 滤波器
series1 = signal.lfilter(b, a, series1)
series2 = signal.lfilter(b, a, series2)

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

ax[0].plot(x, series1, c='b', lw=1)
ax[0].axvline(x=to, c='b', lw=1, ls='--', label=f'x={to}')
ax[0].plot(x, series2+0.1, c='r', lw=1)
ax[0].axvline(x=to+timeshift, c='r', lw=1, ls='--', label=f'x={to+timeshift}')
ax[0].set_yticks([0, 0.1])
ax[0].legend()
ax[0].set_yticklabels(['Trace 1', 'Trace 2'], fontsize=10)

d1, d2 = pd.Series(series2), pd.Series(series1)
lags = np.arange(-(50), (50), 1)

rs = np.nan_to_num([crosscorr(d1, d2, lag) for lag in lags])
maxrs, minrs = np.max(rs), np.min(rs)
if np.abs(maxrs) >= np.abs(minrs):
    corrval = maxrs
else:
    corrval = minrs

ax[1].plot(lags, rs, 'k', label='Xcorr, maxcorr: {:.2f}'.format(corrval), lw=1)
ax[1].axvline(x=lags[np.argmax(rs)], c='r', lw=1, ls='--', label='Correlation')
ax[1].legend(fontsize=8)
plt.subplots_adjust(hspace=0.25, wspace=0.1)

shift = compute_shift(series1, series2)
print(shift)
plt.savefig('Fig 3-9.jpg', dpi=300, bbox_inches='tight')
plt.show()
