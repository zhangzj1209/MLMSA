# Analysis and Association of Vehicle Signals in Multi-stations Seismic Data Based on Artificial Intelligence  

**Copyright (c) 2022 Zhengjie Zhang (zhangzhengjie@mail.ustc.edu.cn)**

by Zhengjie Zhang, Ocean University of China, University of Science and Technology of China  
Email: zhangzhengjie@stu.ouc.edu.cn, zhangzhengjie@mail.ustc.edu.cn  
April 30th, 2022  

This repository is used to store scripts and dataset.  
Limited by the system, this file package only uploads partial data.  


## 1. Installation

### Via Anaconda (recommended):
- Create a new python virtual environment `MLMSA` and activate it
```
conda create -n MLMSA python=3.8
conda activate MLMSA
```

- Install the package
```
conda install numpy==1.21.5 pandas==1.4.2 matplotlib==3.5.1 scipy==1.8.0 obspy==1.3.0
conda install torch==1.11.0           # If you want to install the GPU version, please check the relevant information yourself.
```
If your installation fails, you can try to replace `conda` with `pip`. In addition, you can also try again after replacing *Tsinghua* or *Ustc* source
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
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






* Download repository
* Install dependencies: ```pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt```

The author only provides part of dataset as examples.  
The codes for this study have been run successfully under PyCharm 2021.3.3 (Community Edition).  
* * obspy==1.3.0  
* * matplotlib==3.5.1  
* * numpy==1.21.5  
* * torch==1.11.0  
* * pandas==1.4.2  
* * scipy==1.8.0  



## 2. K-Shape
* data: 50 single-station (EB000207) single-component (Z) data for clustering.

* config_KShape.py: Basic parameters of K-Shape code operation.
* K-Shape.py: K-Shape self-constructing functions and clustering subjects.
* plot_KShape: Drawing related to K-Shape.

* log.txt: Logging of K-Shape results.
* Loss.npy: The record of the loss function.
* NUM_CLU.npy: The record of the number of clusters.



## 3. CNN:
* data: Contains 9,000 sets of training and test data (1500 sets of actual data and 7500 sets of synthetic data),
           150 seconds of application data.
* config_CNN.py: Basic parameters of CNN (Convolutional Neural Networks).
* add_Gaussian_noise.py: Add Gaussian noise to the original 1500 sets of data to expand the data set.
* file_check.py: Inspection of training and testing data.
* generate_label.py: Generate training labels.
* read_SACdata.py: Read each set of SAC data files.
* CNN.py: The module for CNN training and testing.
* predict.py: The module for CNN predicting or applying.
* plot_CNN.py: Drawing related to CNN.

Convert data format to SAC.  
Write seismic station information into "../Code/CNN/data/Station_information.txt"  
Please run "add_Gaussian_noise.py" first, generate a new file and check it with "file_check.py", then use "generate_label.py" to generate labels for training ("CNN.py").  
If you want to add your own data (3-component: E, N, Z), add the training data to '../Code/CNN/data/train_test_data/comprehensive_data'.  
Please regenerate because the labels come with a path.  



## 4. xcorr
* data: Cross-correlation example data.
* config_xcorr.py: Basic parameters of Cross-correlation.
* synthetic_data.py: plot Fig. 3-9.
* synthetic_tests_lib.py: Cross-correlation library packages.
* xcorr.py: Cross-correlation.
