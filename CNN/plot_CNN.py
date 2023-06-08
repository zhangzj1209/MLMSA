import matplotlib.pyplot as plt
import numpy as np
import obspy
import os
from config_CNN import Config

root = Config().root


'''
# 画激活函数
x = np.linspace(-10, 10, 200)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

fig = plt.figure()

# plot sigmoid
ax = fig.add_subplot(221)
ax.plot(x,y_sigmoid,color='k')
plt.xlabel('x')
plt.ylabel('f(x)')
ax.set_title('(a) Sigmoid')

# plot tanh
ax = fig.add_subplot(222)
ax.plot(x,y_tanh,color='k')
plt.xlabel('x')
plt.ylabel('f(x)')
ax.set_title('(b) Tanh')

# plot relu
ax = fig.add_subplot(223)
y_relu = np.array([0*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='k')
plt.xlabel('x')
plt.ylabel('f(x)')
ax.set_title('(c) ReLU')

# plot leaky relu
ax = fig.add_subplot(224)
y_relu = np.array([0.1*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color='k')
plt.xlabel('x')
plt.ylabel('f(x)')
ax.set_title('(d) Leaky ReLU')

plt.tight_layout()
plt.show()
'''


'''
# 画信噪比对比图
path = root + '/data/train_test_data/Associated/'
rawdata_E = obspy.read(path + '220319080303/220319.080310.EB000207.EHE.sac')[0]
rawdata_N = obspy.read(path + '220319080303/220319.080310.EB000207.EHN.sac')[0]
rawdata_Z = obspy.read(path + '220319080303/220319.080310.EB000207.EHZ.sac')[0]
rawdata_E.data = (rawdata_E.data - rawdata_E.data.min()) / (rawdata_E.data.max() - rawdata_E.data.min()) * 2 - 1
rawdata_N.data = (rawdata_N.data - rawdata_N.data.min()) / (rawdata_N.data.max() - rawdata_N.data.min()) * 2 - 1
rawdata_Z.data = (rawdata_Z.data - rawdata_Z.data.min()) / (rawdata_Z.data.max() - rawdata_Z.data.min()) * 2 - 1

newdata_E = obspy.read(path + '200319080303/200319.080303.EB000207.EHE.sac')[0]
newdata_N = obspy.read(path + '200319080303/200319.080303.EB000207.EHN.sac')[0]
newdata_Z = obspy.read(path + '200319080303/200319.080303.EB000207.EHZ.sac')[0]
newdata_E.data = (newdata_E.data - newdata_E.data.min()) / (newdata_E.data.max() - newdata_E.data.min()) * 2 - 1
newdata_N.data = (newdata_N.data - newdata_N.data.min()) / (newdata_N.data.max() - newdata_N.data.min()) * 2 - 1
newdata_Z.data = (newdata_Z.data - newdata_Z.data.min()) / (newdata_Z.data.max() - newdata_Z.data.min()) * 2 - 1

t1 = np.arange(0, rawdata_E.stats.npts / rawdata_E.stats.sampling_rate, rawdata_E.stats.delta)

fig = plt.figure(figsize=(10, 5))
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

ax = fig.add_subplot(311)
ax.plot(t1, rawdata_E.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('E-component', fontsize=14)

ax = fig.add_subplot(312)
ax.plot(t1, rawdata_N.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('N-component', fontsize=14)

ax = fig.add_subplot(313)
ax.plot(t1, rawdata_Z.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('Z-component', fontsize=14)

ax = fig.add_subplot(311)
ax.plot(t1, newdata_E.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('E-component', fontsize=14)

ax = fig.add_subplot(312)
ax.plot(t1, newdata_N.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('N-component', fontsize=14)

ax = fig.add_subplot(313)
ax.plot(t1, newdata_Z.data, 'k')
plt.xlabel('Time(s)', horizontalalignment='right', x=1.0, fontsize=12)
plt.ylabel('Normalized', fontsize=12)
ax.set_title('Z-component', fontsize=14)

plt.tight_layout()
plt.show()
'''


'''
# 画神经网络 Loss 和 Accuracy 图
x = np.linspace(10, 2000, 200)
loss = np.load(root + '/data/train_test_data/train_loss.npy')
accuracy = np.load(root + '/data/train_test_data/train_loss.npy')
plt.plot(x, loss, 'k-', linewidth=2)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
# plt.plot(x, accuracy, 'k-', linewidth=2)
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Accuracy(%)', fontsize=14)
plt.show()
'''


