import time
import matplotlib.pyplot as plt
import numpy as np
import obspy
import os.path
from scipy import fftpack as fp  # Introducing the fast Fourier transform
from scipy import linalg  # Introduction of linear algebraic operations
from config_KShape import Config

config = Config()
root = config.root
lowpass = config.lowpass
new_signal_length = config.new_signal_length

###  Shape-based Distance (SBD)  ###
def get_SBD(x, y):
    # Input: x and y are two z-normalized sequences
    # Output: The difference distance between x and y (Alignment sequence y_shift of y to x)

    # The size of FFT is defined according to the length of the input sequence
    p = int(x.shape[0])
    FFTlen = int(2 ** np.ceil(np.log2(2 * p - 1)))  # Algorithm 1_1

    # Calculate normalized cross-correlation function (NCC)
    CC = fp.ifft(fp.fft(x, FFTlen) * fp.fft(y, FFTlen).conjugate()).real  # Algorithm 1_2  # Equation 12

    # The result of rearranging iFFT
    CC = np.concatenate((CC[-(p - 1):], CC[:p]), axis=0)  # Array splicing

    # Avoid zero division
    denom = linalg.norm(x) * linalg.norm(y)  # linalg.norm(): Calculate the 2-norm
    if denom < 1e-10:
        denom = np.inf
    NCC = CC / denom  # Algorithm 1_3  # Equation 8

    # Find the parameter that maximizes the NCC
    ndx = np.argmax(NCC, axis=0)  # Algorithm 1_4
    dist = 1 - NCC[ndx]  # Algorithm 1_5  # Equation 9
    # Gets the phase shift parameter (if there is no phase shift, s = 0)
    s = ndx - p + 1  # Algorithm 1_6

    # Zero padding according to the shift parameter s
    if s > 0:
        y_shift = np.concatenate((np.zeros(s), y[0:-s]), axis=0)
    elif s == 0:
        y_shift = np.copy(y)  # Algorithm 1_8  # Equation 5
    else:
        y_shift = np.concatenate((y[-s:], np.zeros(-s)), axis=0)  # Algorithm 1_10  # Equation 5

    return dist, y_shift


###  Update k-shape cluster centroid  ###
def shape_extraction(X, C):
    # Input: X is an n×m matrix with z-normalized time series
    #        C is a 1×m reference sequence vector, and the time series of X is aligned with it
    # Output: New_C is a 1×m vector with cluster centroid

    # Define the length of the input
    n = int(X.shape[0])
    m = int(X.shape[1])  # X is a matrix of n×m

    # Construct phase shift signal
    Y = np.zeros((n, m))  # Algorithm 2_1

    for i in range(n):
        # Get the SBD between the centroid and the data
        _, Y[i, :] = get_SBD(C, X[i, :])  # Algorithm 2_2~2_4

    # Constructing the Rayleigh entropy of matrix M
    S = Y.T @ Y  # @ represents the matrix multiplication symbol         # Algorithm 2_5  # S of Equation 15
    Q = np.eye(m) - np.ones((m, m)) / m  # Algorithm 2_6  # Q of Equation 15
    M = Q.T @ (S @ Q)  # Algorithm 2_7  # M of Equation 15

    # The eigenvector corresponding to the maximum eigenvalue is obtained
    eigen_val, eigen_vec = linalg.eig(M)
    ndx = np.argmax(eigen_val, axis=0)
    new_C = eigen_vec[:, ndx].real

    # The ill-posed problem has +C and -C as solutions
    MSE_plus = np.sum((Y - new_C) ** 2)
    MSE_minus = np.sum((Y + new_C) ** 2)
    if MSE_minus < MSE_plus:
        new_C = -1 * new_C

    return new_C


###  Functions for checking empty clusters  ###
def check_empty(label, num_clu):
    # Get a unique label (must include all numbers of 0~num_clu-1)
    label = np.unique(label)

    # Search for empty clusters
    emp_ind = []
    for i in range(num_clu):
        if i not in label:
            emp_ind.append(i)

    # Output the index corresponding to empty cluster
    return emp_ind


###  Get K-shape clustering  ###
def get_KShape(X, num_clu, max_iter, num_init):
    # Input: X is an n×m matrix with a z-normalized time series (containing n sequences of length m)
    #        num_clu is the number of clusters generated
    #        max_iter is iteration times
    #        num_init is test times
    # Output: out_label is an n×1 vector containing n time series assigned to k clusters (random initialization)
    #         out_center is a k×m matrix containing k centroids of length m (initialized as all-zero vectors)
    #         new_loss is the total SBD

    # Define the input length
    n = int(X.shape[0])  # Number of data
    m = int(X.shape[1])  # Length of single data

    # Repeat the experiment (initialization)
    minloss = np.inf
    for init in range(num_init):

        # Initialize the label, centroid and loss into random numbers
        label = np.round((num_clu - 1) * np.random.rand(n))
        center = np.random.rand(num_clu, m)
        loss = np.inf

        # Centroid normalization
        center = center - np.average(center, axis=1)[:, np.newaxis]
        center = center / np.std(center, axis=1)[:, np.newaxis]

        # Temporary copy label
        new_label = np.copy(label)
        new_center = np.copy(center)

        # Repeat each iteration process
        for rep in range(max_iter):

            # Reset loss value
            new_loss = 0

            ###  Refinement steps (update centroid)  ###
            # Repeat each cluster
            for j in range(num_clu):

                # Construct the data matrix of the j-th cluster
                clu_X = []
                for i in range(n):
                    # If the i-th data belongs to the j-th cluster
                    if label[i] == j:
                        clu_X.append(X[i, :])  # Algorithm 3_9
                clu_X = np.array(clu_X)

                # Update cluster centroid
                new_center[j, :] = shape_extraction(clu_X, center[j, :])  # Algorithm 3_10

                # Normalized centroid data
                new_center = new_center - np.average(new_center, axis=1)[:, np.newaxis]
                new_center = new_center / np.std(new_center, axis=1)[:, np.newaxis]

            ###  Allocation step (update the label)  ###
            # Repeat each data
            for i in range(n):

                # Define the minimum distance
                mindist = np.inf

                # Repeat each cluster
                for j in range(num_clu):

                    # Obtain SBD
                    dist, _ = get_SBD(new_center[j, :], X[i, :])

                    # Assign the label corresponding to the minimum distance
                    if dist < mindist:
                        # Update the minimum distance
                        mindist = dist
                        new_label[i] = j

                # Obtain the total amount of SBD
                new_loss = new_loss + mindist

            ###  Error handling (to avoid the emergence of empty clusters)  ###
            # Call our own function to check empty clusters
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    # Assign the same index as the cluster data
                    new_label[ind] = ind

            # If the loss and label remain unchanged, exit the loop
            if loss - new_loss < 1e-6 and (new_label == label).all():
                print("The iteration stopped at {}".format(rep+1))
                break

            # update parameters
            label = np.copy(new_label)
            center = np.copy(new_center)
            loss = np.copy(new_loss)
            print("Loss value: {:.3f}".format(new_loss))

        # Output the results corresponding to the minimum loss function
        if loss < minloss:
            out_label = np.copy(label).astype(np.int16)
            out_center = np.copy(center)
            minloss = loss

    # Output label vector and centroid matrix
    return out_label, out_center, minloss


###  Get the file list  ###
def get_filename(dataset_type):  # dataset_type: The specific folder name (with absolute path)
    filename_dir = os.path.join(root, dataset_type)  # os.path.join: Connect two or more path name components
    if os.path.exists(filename_dir):  # os.path.exists(): To determine whether the file in the brackets exists, the file path can be in the brackets.
        filename_list = os.listdir(filename_dir)  # os.listdir(): Used to return a list of the names of files or folders contained in the specified folder.
        data = list()
        data_length = new_signal_length
        i = 0
        for name in filename_list:
            EventfileDir = os.path.join(filename_dir, name)
            data.append([])
            temp = obspy.read(EventfileDir)[0]
            temp.filter('lowpass', freq=lowpass, corners=2, zerophase=True)  # lowpass filter 
            temp.decimate(factor=2, strict_length=False)
            data[i] = temp.data[:data_length]
            i += 1
    data = np.array(data)

    return data


if __name__ == "__main__":

    time_begin = time.time()

    # optimum configuration
    num_clu = 2  # Number of clusters (default is 2)
    max_iter = 100  # Number of iterations (default is 100)
    num_init = 10  # (Initialization) Number of experiments (default is 10)
    m = 5  # The temporary length or dimension of each data (default is 5)

    # Define random seeds
    np.random.seed(seed=30)

    X = get_filename('data')
    print("Input data shape: {}".format(X.shape))
    # Normalized input data
    X = X - np.average(X, axis=1)[:, np.newaxis]
    X = X / np.std(X, axis=1)[:, np.newaxis]
    ###  clustering procedure  ###
    # Call our own function for K-Shape clustering
    NUM_CLU = np.arange(2, 7)
    Loss = np.empty(5)
    i = 0
    with open(root + '/log.txt', 'a') as f:
        f.write("Input data shape: {}\n".format(X.shape))
    # Save logs
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


    np.save(root + '/NUM_CLU.npy', NUM_CLU)  # Save the number of clusters
    np.save(root + '/Loss.npy', Loss)       # Save the total loss corresponding to each cluster
    plt.plot(NUM_CLU, Loss, marker='o', color='k')
    plt.show()

    time_end = time.time()
    time = time_end - time_begin
    print('time:', time)
