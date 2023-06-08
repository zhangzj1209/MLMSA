import glob
import numpy as np
import obspy
from config_CNN import Config

root = Config().root
signal_length = Config().signal_length

def read_SACdata(filename, Normalize=True):  # file_list即为get_filename函数的返回值
    '''

    :param filename: 文件夹（含绝对路径）
    :param Normalize: 参数归一化
    :return: 返回值data为 [9 ,1500]的数据

    '''

    # 读取台站信息
    station = []
    with open(root + '/data/Station_information.txt', 'r') as f:
        for line in f.readlines():
            if line != '\n':
                if line[:2] != 'EB':
                    station.append([])
                else:
                    station[-1].append(line.split()[0])

    data = np.empty((3 * len(station[0]), 1500))

    for i, sta in enumerate(station[0]):
        sacfile_E = glob.glob(filename + '/' + '*' + sta + '.EHE*')[0]
        sacfile_N = glob.glob(filename + '/' + '*' + sta + '.EHN*')[0]
        sacfile_Z = glob.glob(filename + '/' + '*' + sta + '.EHZ*')[0]
        data_E = obspy.read(sacfile_E)[0]
        data_N = obspy.read(sacfile_N)[0]
        data_Z = obspy.read(sacfile_Z)[0]

        # 归一化
        if Normalize:
            data_E.data = (data_E.data - data_E.data.min()) / (data_E.data.max() - data_E.data.min()) * 2 - 1
            data_N.data = (data_N.data - data_N.data.min()) / (data_N.data.max() - data_N.data.min()) * 2 - 1
            data_Z.data = (data_Z.data - data_Z.data.min()) / (data_Z.data.max() - data_Z.data.min()) * 2 - 1

        data[i * 3][:] = data_E.data[:signal_length]
        data[i * 3 + 1][:] = data_N.data[:signal_length]
        data[i * 3 + 2][:] = data_Z.data[:signal_length]

    return data