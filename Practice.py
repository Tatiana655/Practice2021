import numpy as np
import tensorly.decomposition as decomp
import tensorly as tl
import scipy
import pandas
import matplotlib.pyplot as plt
from matplotlib import cm

def create_Dk(ck):
    '''
    Create diagonal matrix from a vector
    :param ck: 1-way array (1,rank)
    :return:
        Dk: 2-way matrix (rank,rank)
    '''
    Dk = np.zeros((len(ck),len(ck)))
    for i in range(len(ck)):
        Dk[i][i] = ck[i]
    return Dk

def my_parafac(tensor,rank,eps,B,C):
    '''
    Calculate PARAFAC decomposition
    Для реальных данных лучше из коробки взять
    :param tensor: 3-way array
    :param rank: count of component
    :param eps: accuracy
    :param B: second matrix of loadings
    :param C: third matrix of loadings
    :return:
        iter: number of iterations
        A, B, C: matrix of loadings
    '''
    iter = 0
# стоит ещё проверки добавить на входные данные
    if tensor is None:
        return None
    I = len(tensor[0])
    J = len(tensor[0][0])
    K = len(tensor)
    print(rank)
    while (True):
        if iter > 100:
            break
        Z = np.dot(tensor[0], B).dot(create_Dk(C[0]))
        for i in range(1,K,1):
            Z = Z + np.dot(tensor[i], B).dot(create_Dk(C[i]))

        A = Z.dot(np.linalg.inv(np.multiply(C.transpose().dot(C), B.transpose().dot(B))))
        err = scipy.linalg.norm(tl.unfold(tensor, 1) - np.dot(A, np.transpose(scipy.linalg.khatri_rao(C, B))),
                              ord='fro')
        print(err)
        if ( err < eps):
            #table_data.append([eps, iter])
            return iter, A, B, C
        iter += 1
        Z = np.zeros((J, rank))
        for i in range(K):
            Z = Z + np.dot(np.transpose(tensor[i]).dot(A), create_Dk(C[i]))

        B = Z.dot(np.linalg.inv(np.multiply(C.transpose().dot(C), A.transpose().dot(A))))
        for i in range(K):
            tmp1 = np.linalg.inv(np.multiply(B.transpose().dot(B), A.transpose().dot(A)))
            tmp2 = np.dot(np.transpose(A).dot(tensor[i]), B)
            C[i] = np.transpose(np.dot(tmp1.dot(np.multiply(tmp2, np.eye(rank))), np.transpose(np.array([np.ones(rank)]))))

def read_file(name):
    '''
    read data from Bazhenov file.txt || unique func
    :param name: file name
    :return: Ex, Em, Intensity like nparray
    '''
    f = open(name, 'r')

    text = f.read()
    text = text.replace(",",".")
    split_text = text.split("\n")
    for i, str in enumerate(split_text):
        split_text[i] = str.split("\t")
    x_ax = split_text[3][1:len(split_text[3])]
    data = split_text[4:len(split_text)]
    y_ax = []
    for i in range(len(data)):
        y_ax.append(data[i][0])
        data[i]=data[i][1:len(data[i])]

    x_ax = [int(float(x_ax[i])) for i in range(len(x_ax))]
    y_ax = [int(float(y_ax[i])) for i in range(len(y_ax)-1)]
    data = [[float(data[i][j]) for i in range(len(data)-1)] for j in range(len(data[0]))]
    return np.array(x_ax), np.array(y_ax), np.array(data)



def read_tensor(folder_path):
    all_files = glob.glob(folder_path + "*" + txt)  # os.listdir(default_way)
    names = all_files
    print(names)
    tensor_data = []
    for j in range(len(names)):
        g = read_file(names[j])
        tensor_data.append(g[2])
    return g[0],g[1],tensor_data

def NanLine(tuple_matrix, line_width, coef_line, mode = 'zero'):
    '''
    remove Relays lines from data
    :param tuple_matrix: data: ex - 1-way array, em - 1-way array, Intensity - 2-way array
    :param line_width: 1/2 line width (instrument capability)
    :param coef_line: 1 or 2 depending on which line to remove
    :param mode: what to replace - zeros or mean
    :return: new tuple matrix
    '''
    x = tuple_matrix[0]
    y = tuple_matrix[1]
    z = tuple_matrix[2]
    scale = (x[1] - x[0])//(y[1] - y[0]) # масштаб между осями
    for i in range(len(x)):
        const = np.where(y == int(coef_line * x[0]))
        b1 = int(const[0] + scale * i * coef_line - line_width)
        if (b1 < 0):
            b1 =0
        b2 = int(const[0] + scale * i *coef_line + line_width)
        if (b2 > len(y)):
            b2 = len(y)
        if (b1 <= len(y)):
            if (mode == "zero"):
                z[i][b1:b2] = 0
            else:
                if (mode == "Inf"):
                    z[i][b1:b2] = np.Infinity
                else:
                    z[i] = grease_line(z[i],b1,b2-1)
    return x,y,z

def grease_line(line ,b1,b2):
    z1 = 0.
    z2 = 0.
    # b1 = ind[0]
    # b2 = ind[len(ind)-1]
    if (b2 < len(line) - 3):
        z2 = np.mean([line[b2 + 3], line[b2 + 1], line[b2 + 2]])
    if (b1 > 2):
        z1 = np.mean([line[b1 - 3], line[b1 - 1], line[b1 - 2]])
    k = -(z1 - z2) / (b2 - b1+1)
    b = z2 - k * b2
    for h in range(b2 - b1+1):
        line[b1 + h] = k * (b1 + h) + b
    return line

def fill_nan(tuple_matrix):
    x = tuple_matrix[0]
    y = tuple_matrix[1]
    z = tuple_matrix[2]
    for i in range(len(z)):
        ind = np.where(z[i] == np.Infinity)[0]
        if (len(ind)  == 0):
            break
        #найти разделитель, если он есть и в зависимости от этого убирать по разным средним
        dif_array = [(ind[i+1]-ind[i]) for i in range(len(ind)-1)]
        if (len(dif_array)==0):
            z[i][ind[0]] =  z[i][ind[0]-1]
            break
        max_dif = max(dif_array)
        ind_max_dif = np.where(dif_array == max_dif)[0][0]
        if (max_dif <= 5): # смазались
            z[i] = grease_line(z[i],ind[0],ind[len(ind)-1])
        else: #разделены
            z[i] = grease_line(z[i], int(ind[0]), int(ind[ind_max_dif]))
            z[i] = grease_line(z[i], int(ind[ind_max_dif+1]), int(ind[len(ind)-1]))
    return x,y,z

def erase_Reyleigh(data, width1, width2, mode="nzero"):
    x = data[0]
    y = data[1]
    tensor_data = data[2]
    for i in range(len(tensor_data)):
        x,y,tensor_data[i] = NanLine((x,y,tensor_data[i]),width1,1,mode)
        x,y,tensor_data[i] = NanLine((x,y,tensor_data[i]),width2,2,mode)
    return x,y,tensor_data

def erase_Reyleigh_Raman(data, width1, width2, width1_12, width2_2):
    x = data[0]
    y = data[1]
    tensor_data = data[2]
    for i in range(len(tensor_data)):
        x, y, tensor_data[i] = NanLine((x, y, tensor_data[i]), width1, 1, "Inf")
        x, y, tensor_data[i] = NanLine((x, y, tensor_data[i]), width1_12, 1.12, "Inf")
        x, y, tensor_data[i] = fill_nan((x,y,tensor_data[i]))

        x, y, tensor_data[i] = NanLine((x, y, tensor_data[i]), width2, 2, "Inf")
        x, y, tensor_data[i] = NanLine((x, y, tensor_data[i]), width2_2, 2.2, "Inf")
        x, y, tensor_data[i] = fill_nan((x, y, tensor_data[i]))
    return x, y, tensor_data

def show_data(data,h,w,title = "DATA",num1=-1, num2=-1,names = None):
    x = data[0]
    y = data[1]
    tensor_data = data[2]
    if num1 == -1:
        num1 = 0
    if num2 == -1:
        num2 = len(tensor_data)-1
    fig = plt.figure(figsize=(3, 3))
    for j in range(num2-num1+1):
        ax = fig.add_subplot(h, w, j + 1)  # расположения окон
        X, Y = np.meshgrid(x, y)
        if names is not None:
            ax.set_title(names[j])
        cs = ax.contourf(X, Y, np.transpose(tensor_data[num1+j]), levels=100, cmap=cm.coolwarm)
        fig.colorbar(cs)
    plt.suptitle(title)
    plt.show()

def show_loadings(data,rank,tol= 1e-6,iter_max = 1000):
    x = data[0]
    y = data[1]
    tensor_data = data[2]
    print("in process...")
    model = decomp.non_negative_parafac(tensor_data, rank, iter_max, tol=tol)
    fig = plt.figure(figsize=(8, 9))
    plt.suptitle("Rank = " + str(rank), fontsize=18, ha="center")
    ax = fig.add_subplot(3, 1, 3)
    ax.grid()
    ax.plot(np.linspace(1, len(tensor_data) - 1, len(tensor_data) - 1, endpoint=True), model[1][0])

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.grid()
    ax1.plot(x, model[1][1])

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.grid()
    ax2.plot(y, model[1][2])
    plt.show()

def show_components(data,h,w,rank,title = "COMPONENTS",tol= 1e-6,iter_max = 1000):
    x = data[0]
    y = data[1]
    tensor_data = data[2]
    print("in process...")
    model = decomp.non_negative_parafac(tensor_data, rank, iter_max, tol=tol)
    fig = plt.figure(figsize=(8, 5))
    for i in range(rank):
        ax2 = fig.add_subplot(h, w, i + 1)
        mat = np.outer(np.transpose(model[1][1])[i], np.transpose(model[1][2])[i])
        plt.contourf(x, y, np.transpose(mat), levels=100, cmap=cm.coolwarm)
    plt.suptitle(title)
    plt.show()
