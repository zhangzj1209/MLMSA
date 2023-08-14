# Analysis of Vehicle Signals in Multi-stations Seismic Data Based on Artificial Intelligence  

**Copyright (c) 2022 Zhengjie Zhang (zhangzhengjie@mail.ustc.edu.cn)**

- This is the first part of the subject **Analysis and association of vehicle signals in multi-station seismic data based on artificial intelligence**. 
- The second part of the subject belongs to the content of using convolutional neural networks to associate seismic signals. At present, the code is still being sorted out and has not been made public.
- The published data belongs to the test data.

## Description

- **K-Shape:** K-Shape algorithm is used for clustering analysis to find the similarity of signals.
- **xcorr:** The cross-correlation test of the synthetic data and the cross-correlation results of the received signals of the actual stations.
- **ObsPy-Tutorial.pdf:** ObsPy Chinese tutorial -V 1.0 (2020/04/12), you can get more detailed use of ObsPy through the official website https://docs.obspy.org/

## Installation

### Via Anaconda (recommended):
- Create a new python virtual environment `MLMSA` and activate it
```
conda create -n MLMSA python=3.8
conda activate MLMSA
```

- Install the package   
**Note:** We do not recommend that you install the package below the following version, the higher version is adapted, you are welcome to explore more.
```
conda install numpy==1.21.5 pandas==1.4.2 matplotlib==3.5.1 scipy==1.8.0 obspy==1.3.0
```
If your installation fails, you can try to replace `conda` with `pip`. In addition, you can also try again after replacing *Tsinghua* or *Ustc* source
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes
```
or
```
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
conda config --set show_channel_urls yes
```

### Clone source codes
- Set your working directory at `/data/`
```
cd /data/
git clone https://github.com/zhangzj1209/MLMSA.git
unzip MLMSA.zip
cd MLMSA/
```

## K-Shape
```
cd K-Shape/
python K_Shape.py
```
If you want to modify some basic parameters, it will be in `./config_KShape.py`, and the parameters of the clustering algorithm are in the main function part of `./K_Shape.py`.  

The logging of K-Shape results are recorded in `./log.txt`, loss function is recorded in `./Loss.npy`, and the number of clusters is recorded in `./NUM_CLU.npy`.   

Then, you can use `./plot_KShape` to show related figures.

## xcorr
```
cd xcorr/
```
- run the example of synthetic data
```
python synthetic_data.py
```
- run the result of field data
```
python xcorr.py
```
If you want to modify some basic parameters, it will be in `./config_xcorr.py`.
