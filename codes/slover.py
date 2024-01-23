import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import torch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, accuracy_score, f1_score, homogeneity_score
from sklearn.metrics import completeness_score, v_measure_score, davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
from eval import foscttm, acc
from megawass import MegaWass
from filter import g1, g2, g3, g4, g5, g6
import sklearn.decomposition as sk_decomposition

scaler = MinMaxScaler()
def ufgot(X, Y, fiter = 'None', split = False, p1 = [0.1, 0.5, 1, 5, 10, 50, 100], p2 = [0.1, 0.5, 1, 5, 10, 50, 100], eps = 0):
    scaler = MinMaxScaler()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    Y = scaler.fit_transform(Y)
    X = scaler.fit_transform(X)
    X = X.T
    Y = Y.T
    if eps == 0 :
        para_eps = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    else:
        para_eps = [eps]
    if fiter == 'g1':
        X = g1(X)
        Y = g1(Y)
    elif fiter == 'g2':
        X = g2(X)
        Y = g2(Y)
    elif fiter == 'g3':
        X = g3(X)
        Y = g3(Y)
    elif fiter == 'g4':
        X = g4(X)
        Y = g4(Y)
    elif fiter == 'g5':
        X = g5(X)
        Y = g5(Y)
    elif fiter == 'g6':
        X = g6(X)
        Y = g6(Y)
    megawass = MegaWass(nits_bcd=100, nits_uot=1000, tol_bcd=1e-6, tol_uot=1e-6, eval_bcd=1,
                        eval_uot=20)
    if split:
        X_cl = X
        Y_cl = Y
        state = np.random.get_state()
        np.random.shuffle(Y_cl)
        length = len(Y_cl)
        split_length = int(length / 3)

        Y_1 = Y[:split_length]
        Y_2 = Y[split_length:2 * split_length]
        Y_3 = Y[2 * split_length:]
        np.random.set_state(state)
        np.random.shuffle(X_cl)

        X_1 = X[:split_length]
        X_2 = X[split_length:2 * split_length]
        X_3 = X[2 * split_length:]
        X_1 = torch.Tensor(X_1.astype(float)).float().to(device)
        X_2 = torch.Tensor(X_2.astype(float)).float().to(device)
        X_3 = torch.Tensor(X_3.astype(float)).float().to(device)

        Y_1 = torch.Tensor(Y_1.astype(float)).float().to(device)
        Y_2 = torch.Tensor(Y_2.astype(float)).float().to(device)
        Y_3 = torch.Tensor(Y_3.astype(float)).float().to(device)
        i = 1
        eval_ufgot_best = float('inf')
        para_ufgot = []
        for rho1 in p1:
            for rho2 in p2:
                for eps1 in para_eps:
                    rho = (rho1, rho2)
                    eps = (eps1, 0)
                    eval_ufgot = 0
                    X_1_alig = megawass.solver_ufgot(
                        X=X_1,
                        Y=Y_1,
                        rho=rho,
                        eps=eps,
                        log=True,
                        verbose=True,
                        early_stopping_tol=1e-6
                    )
                    X_1_alig = scaler.fit_transform(X_1_alig)
                    eval_ufgot += foscttm(X_1_alig, Y_1)

                    X_2_alig = megawass.solver_ufgot(
                        X=X_2,
                        Y=Y_2,
                        rho=rho,
                        eps=eps,
                        log=True,
                        verbose=True,
                        early_stopping_tol=1e-6
                    )
                    X_2_alig = scaler.fit_transform(X_2_alig)
                    eval_ufgot += foscttm(X_2_alig, Y_2)

                    X_3_alig = megawass.solver_ufgot(
                        X=X_3,
                        Y=Y_3,
                        rho=rho,
                        eps=eps,
                        log=True,
                        verbose=True,
                        early_stopping_tol=1e-6
                    )
                    X_1_alig = scaler.fit_transform(X_1_alig)
                    eval_ufgot += foscttm(X_1_alig, Y_1)

                    if eval_ufgot < eval_ufgot_best:
                        eval_ufgot_best = eval_ufgot
                        para_ufgot = [rho1, rho2, eps1]
                    print(i, '--foscttm:', eval_ufgot/3)
                    i += 1
        X = torch.Tensor(X.astype(float)).float().to(device)

        Y = torch.Tensor(Y.astype(float)).float().to(device)
        data_best_alig = megawass.solver_ufgot(
            X=X,
            Y=Y,
            rho=(para_ufgot[0], para_ufgot[1]),
            eps=(para_ufgot[3], 0),
            log=True,
            verbose=True,
            early_stopping_tol=1e-6
        )
        data_best_alig = scaler.fit_transform(data_best_alig)
    else:
        X = torch.Tensor(X.astype(float)).float().to(device)

        Y = torch.Tensor(Y.astype(float)).float().to(device)
        eval_ufgot_best = float('inf')
        i = 0
        para_ufgot = []
        for rho1 in p1:
            for rho2 in p2:
                for eps1 in para_eps:
                    rho = (rho1, rho2)
                    eps = (eps1, 0)
                    X_alig = megawass.solver_ufgot(
                        X=X,
                        Y=Y,
                        rho=rho,
                        eps=eps,
                        log=True,
                        verbose=True,
                        early_stopping_tol=1e-6
                    )
                    X_alig = scaler.fit_transform(X_alig)
                    eval_ufgot = foscttm(X_alig, Y)
                    print(i, '--foscttm:', eval_ufgot)
                    i += 1
                    if eval_ufgot < eval_ufgot_best:
                        eval_ufgot_best = eval_ufgot
                        data_best_alig = X_alig
    return data_best_alig, Y.cpu().numpy()


cg = sk_decomposition.PCA(whiten=False, svd_solver='auto', n_components=5)

def cluster_truelabel(CNV, Methy, miRNA, mRNA, label, k):
    res = []
    for epo in  range(10):
        totalresult = []

        clu_num = k

        X = np.hstack((CNV, Methy))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 3

        X = np.hstack((CNV, miRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 5

        X = np.hstack((CNV, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 7
        X = np.hstack((Methy, miRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 8

        X = np.hstack((Methy, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 10
        X = np.hstack((miRNA, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])
        # 11
        X = np.hstack((CNV, Methy, miRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])
        X = np.hstack((CNV, Methy, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 9
        X = np.hstack((CNV, miRNA, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 6
        X = np.hstack((Methy, miRNA, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 4
        X = np.hstack((CNV, Methy, miRNA, mRNA))
        spe = KMeans(n_clusters=clu_num).fit(X)
        ind_pre, ind_tru = acc(label, spe.labels_)
        label_pre = spe.labels_
        for i in range(X.shape[0]):
            label_pre[i] = ind_tru[np.where(ind_pre == label_pre[i])]
        totalresult.append(
            [normalized_mutual_info_score(label, spe.labels_), adjusted_mutual_info_score(label, spe.labels_),
             accuracy_score(label, label_pre), f1_score(label, label_pre, average='weighted'),
             homogeneity_score(label, spe.labels_), completeness_score(label, spe.labels_),
             v_measure_score(label, spe.labels_), davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 2
        totalresult = np.array(totalresult)
        if epo == 0:
            res = totalresult
        else:
            res += totalresult
    return res / 10


def cluster_nolabel(CNV, Methy, miRNA, mRNA):
    res = []
    for epo in range(10):
        CNV = cg.fit_transform(CNV)
        Methy = cg.fit_transform(Methy)
        miRNA = cg.fit_transform(miRNA)
        mRNA = cg.fit_transform(mRNA)
        mRNA = cg.fit_transform(mRNA)
        totalresult = []
        X = np.hstack((CNV, Methy))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 3

        X = np.hstack((CNV, miRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 5

        X = np.hstack((CNV, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 7
        X = np.hstack((Methy, miRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 8

        X = np.hstack((Methy, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 10
        X = np.hstack((miRNA, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])
        # 11
        X = np.hstack((CNV, Methy, miRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])
        X = np.hstack((CNV, Methy, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 9
        X = np.hstack((CNV, miRNA, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 6
        X = np.hstack((Methy, miRNA, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        X = cg.fit_transform(X)
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 4
        X = np.hstack((CNV, Methy, miRNA, mRNA))
        models = [GaussianMixture(n_components=i, random_state=233, covariance_type='full').fit(X) for i in range(2, 8)]
        aic = [m.aic(X) for m in models]
        bic = [m.bic(X) for m in models]
        ic = aic + bic
        clu_num = np.where(ic == min(ic))[0] + 2
        spe = KMeans(n_clusters=clu_num[0]).fit(X)
        X = cg.fit_transform(X)
        totalresult.append(
            [davies_bouldin_score(cg.fit_transform(X), spe.labels_),
             calinski_harabasz_score(cg.fit_transform(X), spe.labels_),
             silhouette_score(cg.fit_transform(X), spe.labels_)])  # 2
        totalresult = np.array(totalresult)
        if epo == 0:
            res = totalresult
        else:
            res += totalresult
    return res / 10

def UFGOT_cancer(cancer='COAD', k=0):
    CNV = pd.read_csv('../dataset/' + cancer + '/' + cancer + '_CNV.csv')
    Methy = pd.read_csv('../dataset/' + cancer + '/' + cancer + '_Methy.csv')
    miRNA = pd.read_csv('../dataset/' + cancer + '/' + cancer + '_miRNA.csv')
    mRNA = pd.read_csv('../dataset/' + cancer + '/' + cancer + '_mRNA.csv')
    CNV = CNV.values
    Methy = Methy.values
    miRNA = miRNA.values
    mRNA = mRNA.values
    CNV = np.delete(CNV, 0, axis=1)
    Methy = np.delete(Methy, 0, axis=1)
    miRNA = np.delete(miRNA, 0, axis=1)
    mRNA = np.delete(mRNA, 0, axis=1)
    # train ufgot
    CNV_alig ,y = ufgot(CNV, mRNA, split = True)
    Methy_alig, y = ufgot(Methy, mRNA, split = True)
    miRNA_alig, y = ufgot(miRNA, mRNA, split = True)
    #A set of trained parameters in COAD
    #CNV_alig, y = ufgot(CNV, mRNA, p1=[100], p2=[1], eps = 0.0001)
    #Methy_alig, y = ufgot(Methy, mRNA, p1=[10], p2=[10], eps=0.001)
    #miRNA_alig, y = ufgot(miRNA, mRNA, p1=[1], p2=[1], eps=0.001)
    if k==0:
        res = cluster_nolabel(CNV_alig, Methy_alig, miRNA_alig, mRNA.T)
    else:
        label = pd.read_csv('../dataset/' + cancer + '/' + cancer + '_label.csv')
        label = label.values
        label = label.reshape(label.shape[0],)
        le = LabelEncoder()
        label = le.fit_transform(label)
        res = cluster_truelabel(CNV_alig, Methy_alig, miRNA_alig, mRNA.T, label, k)
    return CNV_alig, Methy_alig, miRNA_alig, y, res
