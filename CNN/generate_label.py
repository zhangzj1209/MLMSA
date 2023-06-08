# 生成数据标签
# 注：标签文件label.txt含路径，请在config_CNN中确定根目录
#    在本文件运行后，请手动对label.txt文件进行操作，删除其最后一行空白行
import os  # 通过os模块调用系统命令
from config_CNN import Config

config = Config()
root = config.root
file_path = root + '/data/train_test_data/comprehensive_data'
file_list = os.listdir(file_path)  # 遍历整个文件夹下的文件name并返回一个列表

with open(root + '/data/train_test_data/label.txt', 'a+') as file:
    file.truncate(0)

path_name = []  # 定义一个空列表

for file_name in file_list:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open(root + '/data/train_test_data/label.txt', "a") as file:
        temp_file_Asso = os.path.join(root, 'data', 'train_test_data', 'Associated', file_name)
        temp_file_NotAsso = os.path.join(root, 'data', 'train_test_data', 'NotAssociated', file_name)
        if os.path.exists(temp_file_Asso):
            file.write(file_path + '/' + file_name + ' 1' + '\n')
        if os.path.exists(temp_file_NotAsso):
            file.write(file_path + '/' + file_name + ' 0' + '\n')