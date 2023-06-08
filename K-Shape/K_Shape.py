import time
import matplotlib.pyplot as plt
import numpy as np
import obspy
import os.path
from scipy import fftpack as fp  # 引入快速Fourier变换
from scipy import linalg  # 引入线性代数

root = 'D:/PyCharm/Code/K-Shape'

###  基于形状的距离测度(Shape-based Distance, SBD)  ###
def get_SBD(x, y):
    # 输入：x和y是两个z-归一化后的序列
    # 输出：x和y的差异性距离
    #      y向x的对齐序列y_shift

    # 根据输入序列的长度定义FFT的大小
    p = int(x.shape[0])
    FFTlen = int(2 ** np.ceil(np.log2(2 * p - 1)))  # Algorithm 1_1

    # 计算归一化互相关函数(normalized cross-correlation function, NCC)
    CC = fp.ifft(fp.fft(x, FFTlen) * fp.fft(y, FFTlen).conjugate()).real  # Algorithm 1_2  # Equation 12

    # 重排iFFT的结果
    CC = np.concatenate((CC[-(p - 1):], CC[:p]), axis=0)  # 数组拼接

    # 避免零除
    denom = linalg.norm(x) * linalg.norm(y)  # linalg.norm(): 计算二范数
    if denom < 1e-10:
        denom = np.inf
    NCC = CC / denom  # Algorithm 1_3  # Equation 8

    # 寻找使NCC最大化的参数
    ndx = np.argmax(NCC, axis=0)  # Algorithm 1_4
    dist = 1 - NCC[ndx]  # Algorithm 1_5  # Equation 9
    # 获取相移参数(如果不存在相移，s=0)
    s = ndx - p + 1  # Algorithm 1_6

    # 根据shift参数s进行零填充
    if s > 0:
        y_shift = np.concatenate((np.zeros(s), y[0:-s]), axis=0)
    elif s == 0:
        y_shift = np.copy(y)  # Algorithm 1_8  # Equation 5
    else:
        y_shift = np.concatenate((y[-s:], np.zeros(-s)), axis=0)  # Algorithm 1_10  # Equation 5

    return dist, y_shift


###  更新k-shape簇质心  ###
def shape_extraction(X, C):
    # 输入：X是一个具有z-归一化的时间序列的n×m的矩阵
    #      C是一个1×m的参考序列向量，X的时间序列与之对齐
    # 输出：new_C是一个含簇质心的1×m的向量

    # 定义输入的长度
    n = int(X.shape[0])
    m = int(X.shape[1])  # X为n×m的矩阵

    # 构造相移信号
    Y = np.zeros((n, m))  # Algorithm 2_1

    for i in range(n):
        # 获取质心和数据之间的SBD
        _, Y[i, :] = get_SBD(C, X[i, :])  # Algorithm 2_2~2_4

    # 构建矩阵M的瑞利熵
    S = Y.T @ Y  # @表示矩阵乘法符号         # Algorithm 2_5  # S of Equation 15
    Q = np.eye(m) - np.ones((m, m)) / m  # Algorithm 2_6  # Q of Equation 15
    M = Q.T @ (S @ Q)  # Algorithm 2_7  # M of Equation 15

    # 得到最大特征值对应的特征向量
    eigen_val, eigen_vec = linalg.eig(M)
    ndx = np.argmax(eigen_val, axis=0)
    new_C = eigen_vec[:, ndx].real

    # 病态问题有+C和-C作为解
    MSE_plus = np.sum((Y - new_C) ** 2)
    MSE_minus = np.sum((Y + new_C) ** 2)
    if MSE_minus < MSE_plus:
        new_C = -1 * new_C

    return new_C


###  用于检查空簇的函数  ###
def check_empty(label, num_clu):
    # 获取唯一标签(必须包括0~ num_clu-1的所有数字)
    label = np.unique(label)

    # 搜寻空簇
    emp_ind = []
    for i in range(num_clu):
        if i not in label:
            emp_ind.append(i)

    # 输出空簇对应的索引
    return emp_ind


###  获取K-shape聚类  ###
def get_KShape(X, num_clu, max_iter, num_init):
    # 输入：X是一个具有z-归一化的时间序列的n×m的矩阵(包含n个长度为m的序列)
    #      num_clu是生成的簇的数量
    #      max_iter是迭代次数
    #      num_init是试验次数
    # 输出：out_label是一个n×1的向量，包含n个时间序列分配到k个簇（随机初始化）
    #      out_center是一个k×m的矩阵，包含k个长度为m的质心（初始化为全零的向量）
    #      new_loss是SBD总量

    # 定义输入长度
    n = int(X.shape[0])  # 数据数量
    m = int(X.shape[1])  # 单个数据长度

    # 重复实验(初始化)
    minloss = np.inf
    for init in range(num_init):

        # 将标签、质心、损失初始化为随机数
        label = np.round((num_clu - 1) * np.random.rand(n))
        center = np.random.rand(num_clu, m)
        loss = np.inf

        # 质心归一化
        center = center - np.average(center, axis=1)[:, np.newaxis]
        center = center / np.std(center, axis=1)[:, np.newaxis]

        # 临时复制标签
        new_label = np.copy(label)
        new_center = np.copy(center)

        # 重复每次迭代过程
        for rep in range(max_iter):

            # 重置损失值
            new_loss = 0

            ###  细化步骤(更新质心)  ###
            # 重复每个簇
            for j in range(num_clu):

                # 构造第j个簇的数据矩阵
                clu_X = []
                for i in range(n):
                    # 如果第i个数据属于第j个簇
                    if label[i] == j:
                        clu_X.append(X[i, :])  # Algorithm 3_9
                clu_X = np.array(clu_X)

                # 更新簇质心
                new_center[j, :] = shape_extraction(clu_X, center[j, :])  # Algorithm 3_10

                # 归一化质心数据
                new_center = new_center - np.average(new_center, axis=1)[:, np.newaxis]
                new_center = new_center / np.std(new_center, axis=1)[:, np.newaxis]

            ###  分配步骤(更新标签)  ###
            # 重复每个数据
            for i in range(n):

                # 定义最小距离
                mindist = np.inf

                # 重复每个簇
                for j in range(num_clu):

                    # 获取SBD
                    dist, _ = get_SBD(new_center[j, :], X[i, :])

                    # 分配对应最小距离的标签
                    if dist < mindist:
                        # 更新最小距离
                        mindist = dist
                        new_label[i] = j

                # 获取SBD总量
                new_loss = new_loss + mindist

            ###  报错处理(避免空簇的出现)  ###
            # 调用自己的函数检查空簇
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    # 分配与簇数据相同的索引
                    new_label[ind] = ind

            # 如果损失和标签不变，则退出循环
            if loss - new_loss < 1e-6 and (new_label == label).all():
                # print("The iteration stopped at {}".format(rep+1))
                break

            # 更新参数
            label = np.copy(new_label)
            center = np.copy(new_center)
            loss = np.copy(new_loss)
            # print("Loss value: {:.3f}".format(new_loss))

        # 输出最小损失函数对应的结果
        if loss < minloss:
            out_label = np.copy(label).astype(np.int16)
            out_center = np.copy(center)
            minloss = loss

    # 输出标签向量和质心矩阵
    return out_label, out_center, minloss


###  获取文件列表  ###
def get_filename(dataset_type):  # dataset_type:具体的文件夹名字(待绝对路径)，如'data'
    filename_dir = os.path.join(root, dataset_type)  # os.path.join:连接两个或更多的路径名组件
    if os.path.exists(filename_dir):  # os.path.exists():判断括号里的文件是否存在，括号内的可以是文件路径
        filename_list = os.listdir(filename_dir)  # os.listdir():用于返回指定的文件夹包含的文件或文件夹的名字的列表
        data = list()
        data_length = 750
        i = 0
        for name in filename_list:
            EventfileDir = os.path.join(filename_dir, name)
            data.append([])
            temp = obspy.read(EventfileDir)[0]
            temp.filter('lowpass', freq=25.0, corners=2, zerophase=True)  # 25Hz低通滤波
            temp.decimate(factor=2, strict_length=False)
            data[i] = temp.data[:data_length]
            i += 1
    data = np.array(data)

    return data


if __name__ == "__main__":

    time_begin = time.time()

    # 参数配置
    # num_clu = 2  # 聚类簇数(默认为2)
    max_iter = 100  # 迭代次数(默认为100)
    num_init = 10  # (初始化)实验次数(默认为10)
    m = 5  # 每个数据的临时长度或维数(默认为5)

    # 定义随机种子
    np.random.seed(seed=30)

    '''
    X = get_filename('data')
    print("Input data shape: {}".format(X.shape))
    # 归一化输入数据
    X = X - np.average(X, axis=1)[:, np.newaxis]
    X = X / np.std(X, axis=1)[:, np.newaxis]

    ###  聚类步骤  ###
    # 调用自己的函数进行 K-Shape 聚类
    label, center, loss = get_KShape(X, num_clu, max_iter, num_init)
    print("Label: {}".format(label))
    print("Centroid: {}".format(center))
    print("Loss: {}".format(loss))

    # 保存日志
    with open(root + '/log.txt', 'a') as f:
        f.write("Input data shape: {}\nLabel: {}\nLoss: {}".format(X.shape, label, loss))
    '''


    X = get_filename('data')
    print("Input data shape: {}".format(X.shape))
    # 归一化输入数据
    X = X - np.average(X, axis=1)[:, np.newaxis]
    X = X / np.std(X, axis=1)[:, np.newaxis]
    ###  聚类步骤  ###
    # 调用自己的函数进行 K-Shape 聚类
    NUM_CLU = np.arange(2, 7)
    Loss = np.empty(5)
    i = 0
    with open(root + '/log.txt', 'a') as f:
        f.write("Input data shape: {}\n".format(X.shape))
    # 保存日志
    for num_clu in NUM_CLU:
        print("Now Number of Clustering: {}".format(num_clu))
        label, center, loss = get_KShape(X, num_clu, max_iter, num_init)
        print("Label: {}".format(label))
        print("Centroid: {}".format(center))
        print("Loss: {}\n".format(loss))
        with open(root + '/log.txt', 'a') as f:
            f.write("Label: {}\nLoss: {}\n\n".format(label, loss))
        Loss[i] = loss
        i = i + 1


    np.save(root + '/NUM_CLU.npy', NUM_CLU)  # 保存簇数
    np.save(root + '/Loss.npy', Loss)       # 保存各簇对应的总损失
    plt.plot(NUM_CLU, Loss, marker='o', color='k')
    plt.show()

    time_end = time.time()
    time = time_end - time_begin
    print('time:', time)


