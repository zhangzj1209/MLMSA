import time
import numpy as np
import obspy
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from read_SACdata import read_SACdata
from config_CNN import Config

num_epochs = Config().epoch     # epoch
batch_size = Config().batch_size    # batch_size
root = Config().root        # root
label_txt = root + '/data/train_test_data/label.txt'    # 标签txt文件


class MySignalDataset(Dataset):
    def __init__(self, txt_path):
        fh = open(txt_path, 'r')
        signals = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            signals.append((words[0], int(words[1])))
            self.signals = signals

    def __getitem__(self, index):
        fn, label = self.signals[index]
        signal = read_SACdata(fn, Normalize=True)

        return signal, label

    def __len__(self):
        return len(self.signals)


all_dataset = MySignalDataset(label_txt)    # 总数据集
train_size = int(len(all_dataset) * 0.8)    # 训练集
test_size = len(all_dataset) - train_size   # 测试集
train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_num_batches = len(test_loader)

'''
num_epochs = Config().epoch
batch_size = Config().batch_size
root = Config().root
label_txt = root + '/data/train_test_data/label.txt'
train_dataset = MySignalDataset(label_txt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MySignalDataset(label_txt)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=1, padding=2),  # b, 32, 1, 1500
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0)  # b, 32, 1, 750
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),  # b, 64(channels), 1, 750
            nn.BatchNorm1d(64),  # 归一化层
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)  # b, 64, 1, 250
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0),  # b, 128, 1, 246
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)  # b, 128, 1, 82
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),  # b, 256, 1, 78
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)  # b, 256, 1, 26
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1),  # b, 512, 1, 24
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0)  # b, 512, 1, 8
        )

        self.fc1 = nn.Linear(512 * 8, 100)  # (input, output)  # Linear是全连接层Fully Connected
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)  # (input, output)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)  # (input, output)

    def forward(self, x):
        x = self.layer1(x)  # (batch, 32, 1, 750)
        x = self.layer2(x)  # (batch, 64, 1, 250) -> (batch_size, 输出channels, 输出height, 输出width)
        x = self.layer3(x)  # (batch, 128, 1, 82)
        x = self.layer4(x)  # (batch, 256, 1, 26)
        x = self.layer5(x)  # (batch, 512, 1, 8)

        x = x.view(x.size(0), -1)  # 扩展、展平 -> (batch, 512 * 1 * 8)   # view是将多维数据平铺为一维
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = x.squeeze(-1)
        return x


def train():
    model = CNN()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()中已经包含Softmax，网络端为线性输出

    train_loss, train_acc = [], []  # 用于记录保存损失函数变化、精度变化（每10次迭代记录一次）

    for epoch in range(num_epochs):
        running_loss, running_acc = 0.0, 0.0
        loss_epoch, acc_epoch = 0.0, 0.0    # 每个epoch的平均损失、平均精度
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            inputs = inputs.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_epoch += loss.item()   # 每个epoch的平均损失记录

            _, predict = torch.max(outputs, 1)
            correct_num = (predict == target).sum()
            running_acc += correct_num.data
            acc_epoch += correct_num.data    # 每个epoch的平均精度记录

            if batch_idx % 10 == 9:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}%'.format(epoch+1,
                                                                                                     num_epochs,
                                                                                                     batch_idx*len(inputs),
                                                                                                     len(train_loader.dataset),
                                                                                                     100.*batch_idx*len(inputs)/len(train_loader.dataset),
                                                                                                     running_loss/10,
                                                                                                     100*running_acc/(10*batch_size)))

                train_loss.append(running_loss/10)
                train_acc.append(100*running_acc/(10*batch_size))

                running_loss = 0.0
                running_acc = 0.0

        loss_epoch = loss_epoch / (len(train_dataset) / batch_size)  # 每个epoch的平均损失
        acc_epoch = acc_epoch / len(train_dataset) * 100  # 每个epoch的平均精度

        print('\n------------------------------------------------------------------------------------------------')
        print('{}/{}\tAverage Loss: {:.6f}\tAverage Accuracy: {:.4f}%'.format(epoch+1, num_epochs, loss_epoch, acc_epoch))
        print('------------------------------------------------------------------------------------------------\n')

    # 每10次迭代的损失和精度保存（用于后期成图）
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    np.save(root + '/data/train_test_data/train_loss.npy', train_loss)
    np.save(root + '/data/train_test_data/train_acc.npy', train_acc)

    print('Finished Training!')
    torch.save(model, root + '/data/train_test_data/model.pkl')
    torch.save(model.state_dict(), root + '/data/train_test_data/model_params.pkl')


def reload_net():
    trainednet = torch.load(root + '/data/train_test_data/model.pkl')
    return trainednet


def test():
    model = reload_net()
    model.eval()
    criterion = nn.CrossEntropyLoss()

    TP, TN, FP, FN = 0, 0, 0, 0
    epsilon = 1e-6

    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            signal, label = data

            signal = signal.to(torch.float32)
            outputs = model(signal)

            _, predicted = torch.max(outputs.data, dim=1)
            if predicted == 0 and label == 0:
                TP += 1
            if predicted == 1 and label == 1:
                TN += 1
            if predicted == 0 and label == 1:
                FP += 1
            if predicted == 1 and label == 0:
                FN += 1

            test_loss += criterion(outputs, label).item()

    test_loss /= test_num_batches
    Precision = TP / (TP + FP + epsilon)
    Recall = TP / (TP + FN + epsilon)
    F1 = 2 * Precision * Recall / (Precision + Recall + epsilon)
    Accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

    print('Summary of Test Results:')
    print('Precision: {:.2f} %'.format(100*Precision))
    print('Recall: {:.2f} %'.format(100 * Recall))
    print('F1: {:.2f} %'.format(100 * F1))
    print('Accuracy: {:.2f} %'.format(100 * Accuracy))
    print('Loss: {} '.format(test_loss))

    '''
    # 无需计算Precision, Recall, F1等
    test_loss, correct = 0, 0
    with torch.no_grad():  # 使得以下代码执行过程中不用求梯度
        correct = 0
        total = 0
        test_loss = 0.0
        for data in test_loader:
            signal, label = data
            signal = signal.to(torch.float32)
            outputs = model(signal)
            # 在每一行中求最大值的下标，返回两个参数，第一个为最大值，第二个为坐标
            # dim=1 数值方向的维度为0，水平方向的维度为1

            loss = criterion(outputs, label)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += label.size(0)
            # 当预测值与标签相同时取出并求和
            correct += (predicted == label).sum().item()

    print('Accurary on test set: %f %%' % (100*correct/total))
    print('Loss on test set: %f' % (test_loss / (len(train_dataset) / batch_size)))
    '''

def predict(input):
    model = reload_net()
    with torch.no_grad():
        out = model(input)
        _, pre = torch.max(out.data, 1)
        return pre.item()


if __name__ == '__main__':
    begin_time = time.time()
    train()  # 训练
    test()   # 测试

    '''
    sampling = 250
    total_time = 150
    total_data_length = sampling * total_time
    window = 6
    step = 4

    path = 'C:\\Users\\Zhengjie Zhang\\Desktop\\test\\'
    data = np.empty((9, 1500))
    data[0, :] = obspy.read(path + '220320.080510.EB000203.EHE.sac')[0].data
    data[1, :] = obspy.read(path + '220320.080510.EB000203.EHN.sac')[0].data
    data[2, :] = obspy.read(path + '220320.080510.EB000203.EHZ.sac')[0].data
    data[3, :] = obspy.read(path + '220320.080515.EB000207.EHE.sac')[0].data
    data[4, :] = obspy.read(path + '220320.080515.EB000207.EHN.sac')[0].data
    data[5, :] = obspy.read(path + '220320.080515.EB000207.EHZ.sac')[0].data
    data[6, :] = obspy.read(path + '220320.080513.EB000208.EHE.sac')[0].data
    data[7, :] = obspy.read(path + '220320.080513.EB000208.EHN.sac')[0].data
    data[8, :] = obspy.read(path + '220320.080513.EB000208.EHZ.sac')[0].data
    data = torch.from_numpy(data)
    data = data.to(torch.float32)
    data = data.unsqueeze_(dim=0)
    print(predict(data))
    '''
    '''
    path = 'D:/PyCharm/PyCharmProjects/pythonProject/datasets/CNN/datasets/application_data/test_0510/'

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

    num_car = 0
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
                # num_car += 1
                break
        if result == 1:
            # result = 0
            break

    print(j, k)
    print('Number of Car:', num_car)
    now_time = time.time()
    print('Running Time: ', now_time - begin_time, 's')
    '''





    '''
    temp = np.empty((3, 30000))
    temp[0, :] = obspy.read(path + '220320.080312.EB000203.EHE.sac')[0].data[:total_data_length]
    temp[1, :] = obspy.read(path + '220320.080312.EB000203.EHN.sac')[0].data[:total_data_length]
    temp[2, :] = obspy.read(path + '220320.080312.EB000203.EHZ.sac')[0].data[:total_data_length]
    # for i in range(3):
        # temp[i, :] = (temp[i, :] - temp[i, :].min()) / (temp[i, :].max() - temp[i, :].min()) * 2 - 1
    data_EB000203 = temp

    temp = np.empty((3, 30000))
    temp[0, :] = obspy.read(path + '220320.080312.EB000207.EHE.sac')[0].data[:total_data_length]
    temp[1, :] = obspy.read(path + '220320.080312.EB000207.EHN.sac')[0].data[:total_data_length]
    temp[2, :] = obspy.read(path + '220320.080312.EB000207.EHZ.sac')[0].data[:total_data_length]
    # for i in range(3):
    #     temp[i, :] = (temp[i, :] - temp[i, :].min()) / (temp[i, :].max() - temp[i, :].min()) * 2 - 1
    data_EB000207 = temp

    temp = np.empty((3, 30000))
    temp[0, :] = obspy.read(path + '220320.080312.EB000208.EHE.sac')[0].data[:total_data_length]
    temp[1, :] = obspy.read(path + '220320.080312.EB000208.EHN.sac')[0].data[:total_data_length]
    temp[2, :] = obspy.read(path + '220320.080312.EB000208.EHZ.sac')[0].data[:total_data_length]
    # for i in range(3):
        # temp[i, :] = (temp[i, :] - temp[i, :].min()) / (temp[i, :].max() - temp[i, :].min()) * 2 - 1
    data_EB000208 = temp

    window_EB000203 = np.lib.stride_tricks.sliding_window_view(data_EB000203, (3, sampling * window)).squeeze()[::sampling * step, :, :]
    window_EB000207 = np.lib.stride_tricks.sliding_window_view(data_EB000207, (3, sampling * window)).squeeze()[::sampling * step, :, :]
    window_EB000208 = np.lib.stride_tricks.sliding_window_view(data_EB000208, (3, sampling * window)).squeeze()[::sampling * step, :, :]

    num_window = len(window_EB000203)  # num_window = 39

    # print(window_EB000203.shape)
    # print(type(window_EB000207[0][2]))
    num_car = 0
    # Test_data = np.concatenate((window_EB000203[0], window_EB000208[0], window_EB000207[0]), axis=0)
    # print(Test_data)
    # print(Test_data.shape)

    # plt.subplots(311)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(311)
    ax.plot(window_EB000207[1][2], 'k')

    ax = fig.add_subplot(312)
    ax.plot(window_EB000208[1][2], 'k')

    ax = fig.add_subplot(313)
    ax.plot(window_EB000203[1][2], 'k')

    plt.show()

    result = 0
    for i in range(num_window):  # 本次实验以208为基础
        for j in range(i - 5, i + 5):
            for k in range(j - 5, j + 5):
                if j >= 0 and k >= 0 and j < num_window and k < num_window:
                    Test_data = np.concatenate((window_EB000208[i], window_EB000203[j], window_EB000207[k]), axis=0)

                    Test_data = (Test_data - np.min(Test_data, axis=1)[:, np.newaxis]) / (
                                np.max(Test_data, axis=1)[:, np.newaxis] - np.min(Test_data, axis=1)[:, np.newaxis])

                    Test_data = torch.from_numpy(Test_data)
                    Test_data = Test_data.to(torch.float32)
                    Test_data = Test_data.unsqueeze_(dim=0)
                    result = predict(Test_data)
                    if result == 1:
                        num_car += 1
                        break
            if result == 1:
                result = 0
                break
            # if result == 1:
            #     break
        # if result == 1:
        #     break
    print(i, j, k)
    print('Number of Car:', num_car)

    '''

