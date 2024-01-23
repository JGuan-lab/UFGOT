import numpy as np
import pandas as pd
from scipy import ndimage
import scipy as sp
import matplotlib.pylab as pl
import scipy.io
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, accuracy_score, f1_score, homogeneity_score
from sklearn.metrics import completeness_score, v_measure_score, davies_bouldin_score, silhouette_score, calinski_harabasz_score
import torch
from sklearn.preprocessing import MinMaxScaler
import sys


def foscttm(
        x: np.ndarray, y: np.ndarray, **kwargs
) :
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return ((foscttm_x + foscttm_y)/2).mean()


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_row, ind_clo = linear_sum_assignment(w.max() - w)
    #return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind
    return ind_row, ind_clo