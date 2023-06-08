# 添加高斯噪声扩充数据集
# 注：添加高斯噪声后的数据通过改变文件名而存在，即通过修改 SNR 和 temp 完成
#     在本文件运行后，请手动将合成数据添加到comprehensive_data中，再生成生成标签，运行generate_label
import numpy as np
import obspy
import os
import glob
from config_CNN import Config

def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR/10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return noise

def check_SNR(signal, noise):
    '''
    :param signal: 原始信号
    :param noise: 生成的高斯噪声
    :return: 返回两者的信噪比
    '''
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))
    SNR = 10*np.log10(signal_power / noise_power)
    return SNR


SNR = 1     # 信噪比，此参数需要改变来生成不同的合成数据集
temp = 50000000000      # 命名方式，此参数需要改变来命名新的合成数据集

root = Config().root
path_original_file = root + '/data/train_test_data/Associated'  # Associated 或 NotAssociated
path_new_file = root + '/data/train_test_data/temp'     # 请将添加高斯噪声生成后的数据在临时文件夹中暂存，最终一并移入 Associated 或 NotAssociated 组成新数据集
filename_list = os.listdir(path_original_file)

for name in filename_list:
    EventfileDir = path_original_file + '/' + name
    os.chdir(EventfileDir)
    sacfile_EB203_E = glob.glob(EventfileDir + '/' + '*' + 'EB000203.EHE*')[0]
    sacfile_EB203_N = glob.glob(EventfileDir + '/' + '*' + 'EB000203.EHN*')[0]
    sacfile_EB203_Z = glob.glob(EventfileDir + '/' + '*' + 'EB000203.EHZ*')[0]
    sacfile_EB207_E = glob.glob(EventfileDir + '/' + '*' + 'EB000207.EHE*')[0]
    sacfile_EB207_N = glob.glob(EventfileDir + '/' + '*' + 'EB000207.EHN*')[0]
    sacfile_EB207_Z = glob.glob(EventfileDir + '/' + '*' + 'EB000207.EHZ*')[0]
    sacfile_EB208_E = glob.glob(EventfileDir + '/' + '*' + 'EB000208.EHE*')[0]
    sacfile_EB208_N = glob.glob(EventfileDir + '/' + '*' + 'EB000208.EHN*')[0]
    sacfile_EB208_Z = glob.glob(EventfileDir + '/' + '*' + 'EB000208.EHZ*')[0]

    data_EB203_E = obspy.read(sacfile_EB203_E)[0]
    data_EB203_N = obspy.read(sacfile_EB203_N)[0]
    data_EB203_Z = obspy.read(sacfile_EB203_Z)[0]
    data_EB207_E = obspy.read(sacfile_EB207_E)[0]
    data_EB207_N = obspy.read(sacfile_EB207_N)[0]
    data_EB207_Z = obspy.read(sacfile_EB207_Z)[0]
    data_EB208_E = obspy.read(sacfile_EB208_E)[0]
    data_EB208_N = obspy.read(sacfile_EB208_N)[0]
    data_EB208_Z = obspy.read(sacfile_EB208_Z)[0]

    data_EB203_E.data = data_EB203_E.data + gen_gaussian_noise(data_EB203_E.data, SNR)
    data_EB203_N.data = data_EB203_N.data + gen_gaussian_noise(data_EB203_N.data, SNR)
    data_EB203_Z.data = data_EB203_Z.data + gen_gaussian_noise(data_EB203_Z.data, SNR)
    data_EB207_E.data = data_EB207_E.data + gen_gaussian_noise(data_EB207_E.data, SNR)
    data_EB207_N.data = data_EB207_N.data + gen_gaussian_noise(data_EB207_N.data, SNR)
    data_EB207_Z.data = data_EB207_Z.data + gen_gaussian_noise(data_EB207_Z.data, SNR)
    data_EB208_E.data = data_EB208_E.data + gen_gaussian_noise(data_EB208_E.data, SNR)
    data_EB208_N.data = data_EB208_N.data + gen_gaussian_noise(data_EB208_N.data, SNR)
    data_EB208_Z.data = data_EB208_Z.data + gen_gaussian_noise(data_EB208_Z.data, SNR)

    os.chdir(path_new_file)
    newfilefolder = str(int(name) - temp)
    os.mkdir(newfilefolder)
    os.chdir(newfilefolder)
    x = str(round(float(newfilefolder) * 1e-6, 6))

    data_EB203_E.write(x + '.' + 'EB000203.EHE.sac', format="SAC")
    data_EB203_N.write(x + '.' + 'EB000203.EHN.sac', format="SAC")
    data_EB203_Z.write(x + '.' + 'EB000203.EHZ.sac', format="SAC")
    data_EB207_E.write(x + '.' + 'EB000207.EHE.sac', format="SAC")
    data_EB207_N.write(x + '.' + 'EB000207.EHN.sac', format="SAC")
    data_EB207_Z.write(x + '.' + 'EB000207.EHZ.sac', format="SAC")
    data_EB208_E.write(x + '.' + 'EB000208.EHE.sac', format="SAC")
    data_EB208_N.write(x + '.' + 'EB000208.EHN.sac', format="SAC")
    data_EB208_Z.write(x + '.' + 'EB000208.EHZ.sac', format="SAC")


