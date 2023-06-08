# 检查文件夹中的文件是否正确
import os
from config_CNN import Config

root = root = Config().root
path = root + '/data/train_test_data/comprehensive_data'
file_list = os.listdir(path)
for name in file_list:
    count = 0
    data = os.path.join(path, name)

    for file in os.listdir(data):
        count = count + 1

    if count != 9:
        print(name)
        break