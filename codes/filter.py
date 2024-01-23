import numpy as np
from numpy import linalg as la
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math


def g1(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.sqrt(sigma[i])
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


def g2(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.pow(sigma[i], 2)
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


def g3(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.pow(math.e, -0.8*sigma[i])
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


def g4(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.pow(sigma[i], 2) + math.sqrt(sigma[i])
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


def g5(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.pow(math.e, -0.8*sigma[i]) + math.sqrt(sigma[i])
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


def g6(L):
    std = MinMaxScaler()
    u, sigma, vt = la.svd(L)
    S = np.zeros((u.shape[0], vt.shape[0]))
    for i in range(sigma.shape[0]):
        S[i][i] = math.pow(math.e, -0.8*sigma[i]) + math.pow(sigma[i], 2)
    L_fiter = u @ S @ vt
    L_fiter = std.fit_transform(L_fiter)
    return L_fiter


