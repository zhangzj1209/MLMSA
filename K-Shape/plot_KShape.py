import matplotlib.pyplot as plt
import numpy as np
import obspy
from scipy.fftpack import fft,fftshift
from config_KShape import Config

root = Config().root
lowpass = Config().lowpass
sample = Config().sample
resample = Config().resample
original_signal_length = Config().original_signal_length
new_signal_length = Config().new_signal_length

'''
# 画原始波形、滤波波形、重采样波形及单边频谱图
file = root + '/data/' + '220318.191012.EB000207.EHZ.sac'

fig = plt.figure(figsize=(12, 6))
plt.rcParams['axes.unicode_minus'] = False

rawdata = obspy.read(file)[0]
rawdata.data = (rawdata.data - rawdata.data.min()) / (rawdata.data.max() - rawdata.data.min()) * 2 - 1
t1 = np.arange(0, rawdata.stats.npts / rawdata.stats.sampling_rate, rawdata.stats.delta)
fft_rawdata = fft(rawdata)  # FFT变换
fft_rawdata_amp0 = np.array(np.abs(fft_rawdata)/original_signal_length*2)
fft_rawdata_amp0[0] = 0.5 * fft_rawdata_amp0[0]
fft_rawdata_amp1 = fft_rawdata_amp0[0: int(original_signal_length/2)]   # 单边谱

# 计算频谱的频率轴
list0 = np.array(range(0, original_signal_length))
list1 = np.array(range(0, int(original_signal_length/2)))
freq0 = sample * list0 / original_signal_length     # 双边谱的频率轴
freq1 = sample * list1 / original_signal_length     # 单边谱的频率轴

ax = fig.add_subplot(321)
ax.plot(t1, rawdata.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
plt.ylim(-1.2, 1.2)
ax.set_title('(a) Original Waveform', fontsize=14)

ax = fig.add_subplot(322)
ax.plot(freq1[1:300], fft_rawdata_amp1[1:300], 'k')
plt.xlabel('Frequency(Hz)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.ylim(0, 0.06)
ax.set_title('(b) Spectrum of Original Waveform', fontsize=14)

# 25Hz低通滤波
rawdata = rawdata.filter('lowpass', freq=lowpass, corners=2, zerophase=True)    # 25Hz低通滤波
# rawdata.data = (rawdata.data - rawdata.data.min()) / (rawdata.data.max() - rawdata.data.min()) * 2 - 1
fft_rawdata = fft(rawdata)  # FFT变换
fft_rawdata_amp0 = np.array(np.abs(fft_rawdata)/original_signal_length*2)
fft_rawdata_amp0[0] = 0.5 * fft_rawdata_amp0[0]
fft_rawdata_amp1 = fft_rawdata_amp0[0: int(original_signal_length/2)]   # 单边谱

# 计算频谱的频率轴
list0 = np.array(range(0, original_signal_length))
list1 = np.array(range(0, int(original_signal_length/2)))
freq0 = sample * list0 / original_signal_length     # 双边谱的频率轴
freq1 = sample * list1 / original_signal_length     # 单边谱的频率轴

ax = fig.add_subplot(323)
ax.plot(t1, rawdata.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
plt.ylim(-1.2, 1.2)
ax.set_title('(c) Lowpass Filter Waveform', fontsize=14)

ax = fig.add_subplot(324)
ax.plot(freq1[1:300], fft_rawdata_amp1[1:300], 'k')
plt.xlabel('Frequency(Hz)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.ylim(0, 0.06)
ax.set_title('(d) Spectrum of Lowpass Filter Waveform', fontsize=14)

# 重采样
rawdata = rawdata.decimate(factor=2, strict_length=False)
t2 = np.arange(0, rawdata.stats.npts / rawdata.stats.sampling_rate, rawdata.stats.delta)
# rawdata.data = (rawdata.data - rawdata.data.min()) / (rawdata.data.max() - rawdata.data.min()) * 2 - 1
fft_rawdata = fft(rawdata)  # FFT变换
fft_rawdata_amp0 = np.array(np.abs(fft_rawdata)/new_signal_length*2)
fft_rawdata_amp0[0] = 0.5 * fft_rawdata_amp0[0]
fft_rawdata_amp1 = fft_rawdata_amp0[0: int(new_signal_length/2)]   # 单边谱

# 计算频谱的频率轴
list0 = np.array(range(0, new_signal_length))
list1 = np.array(range(0, int(new_signal_length/2)))
freq0 = resample * list0 / new_signal_length     # 双边谱的频率轴
freq1 = resample * list1 / new_signal_length     # 单边谱的频率轴

ax = fig.add_subplot(325)
ax.plot(t2, rawdata.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
plt.ylim(-1.2, 1.2)
ax.set_title('(e) Resampling Waveform', fontsize=14)

ax = fig.add_subplot(326)
ax.plot(freq1[1:300], fft_rawdata_amp1[1:300], 'k')
plt.xlabel('Frequency(Hz)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.ylim(0, 0.06)
ax.set_title('(f) Spectrum of Resampling Waveform', fontsize=14)

plt.tight_layout()
plt.show()
'''


'''
# 画 K-Shape 聚类的肘部法则结果图
Loss = np.load(root + '/Loss.npy')
NUM_CLU = np.load(root + '/NUM_CLU.npy')
plt.plot(NUM_CLU, Loss, marker='o', color='k')
plt.ylim(16, 24)
plt.xlabel('K', horizontalalignment='right', x=1.0, fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.tick_params(labelsize=12)
plt.title('The Elbow Method', fontsize=15)
plt.show()
'''

