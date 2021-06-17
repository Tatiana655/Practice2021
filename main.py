import numpy as np
import tensorly.decomposition as decomp
import tensorly as tl
import scipy
import pandas
import matplotlib.pyplot as plt
from matplotlib import cm

tensor = np.array([
  [ [1., 1., 0.],
    [1., 1., 0.],
    [0., 0., 0.] ],
  [ [0., 0., 0.],
    [0., 1., 1.],
    [0., 1., 1.] ],
  [ [0.5, 0.5, 0.],
    [0.5, 1., 0.5],
    [0., 0.5, 0.5] ]
])

tensor1 = np.array([
  [ 1., 1., 0.,
    1., 1., 0.,
    0., 0., 0. ],
  [ 0., 0., 0.,
    0., 1., 1.,
    0., 1., 1. ],
  [ 0.5, 0.5, 0.,
    0.5, 1., 0.5,
    0., 0.5, 0.5 ]
])

KoldaTensor = np.array(
    [
        [1.,0., 0.,1.],
        [0.,1.,-1.,0.]
    ]
)

default_way = "C:\\Users\\Tatiana\\Desktop\\ToBazhenov\\VD_DOM_Permafrost\\"
txt = '.txt'


def create_Dk(ck):
    '''
    Create diagonal matrix from a vector
    :param ck: 1-way array (1,rank)
    :return:
        Dk: 2-way matrix (rank,rank)
    '''
    Dk = np.zeros((2,2))
    for i in range(2):
        Dk[i][i] = ck[i]
    return Dk

def my_parafac(tensor,rank,eps,B,C):
    '''
    Calculate PARAFAC decomposition
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

    while (True):
        Z = np.zeros((I, rank))
        for i in range(I):
            Z = Z + np.dot(tensor[i], B).dot(create_Dk(C[i]))

        A = Z.dot(np.linalg.inv(np.multiply(C.transpose().dot(C), B.transpose().dot(B))))

        if (scipy.linalg.norm(tl.unfold(tensor, 1) - np.dot(A, np.transpose(scipy.linalg.khatri_rao(C, B))),
                              ord='fro') < eps):
            #table_data.append([eps, iter])
            return iter, A, B, C
        iter += 1
        Z = np.zeros((J, rank))
        for i in range(J):
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
        const = np.where(y == coef_line * x[0])
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
                z1 = 0.
                z2 = 0.
                if (b2 < len(y)-2):
                    z2 = np.mean([z[i][b2], z[i][b2+1], z[i][b2+2]])
                if (b1 > 2):
                    z1 = np.mean([z[i][b1-3], z[i][b1-1], z[i][b1-2]])
                k = -(z1-z2)/(b2-b1)
                b = z2 - k*b2
                for h in range(b2-b1):
                    z[i][b1 + h] = k*(b1 + h)+b
    return x,y,z

