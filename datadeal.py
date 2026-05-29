import numpy as np
import math
import pandas as pd

# ---------------------------------------------------------
# 相似性计算工具函数
# ---------------------------------------------------------

def Cosine_Sim(M):
    """余弦相似性"""
    l = len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1 = np.dot(M[i], M[j])
            v2 = np.linalg.norm(M[i], ord=2)
            v3 = np.linalg.norm(M[j], ord=2)
            if v2 * v3 == 0:
                SM[i][j] = 0
            else:
                SM[i][j] = v1 / (v2 * v3)
    return SM

def HIP_Calculate(M):
    """Hamming Interaction Profile 相似性"""
    l = len(M)
    cl = np.size(M, axis=1)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k] != M[j][k]:
                    dnum = dnum + 1
            SM[i][j] = 1 - dnum / cl
    return SM

def GIP_Calculate(M):
    """高斯核相似性 (通用/微生物)"""
    l = np.size(M, axis=1)
    sm = []
    m = np.zeros((l, l))
    for i in range(l):
        tmp = (np.linalg.norm(M[:, i])) ** 2
        sm.append(tmp)
    if np.sum(sm) == 0:
        return m
    gama = l / np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i, j] = np.exp(-gama * ((np.linalg.norm(M[:, i] - M[:, j])) ** 2))
    return m

def GIP_Calculate1(M):
    """高斯核相似性 (药物) - 带Log平滑"""
    l = np.size(M, axis=0)
    sm = []
    m = np.zeros((l, l))
    km = np.zeros((l, l))
    for i in range(l):
        tmp = (np.linalg.norm(M[i, :])) ** 2
        sm.append(tmp)
    if np.sum(sm) == 0:
        return km
    gama = l / np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i, j] = np.exp(-gama * ((np.linalg.norm(M[i, :] - M[j, :])) ** 2))
    for i in range(l):
        for j in range(l):
            km[i, j] = 1 / (1 + np.exp(-15 * m[i, j] + math.log(9999)))
    return km

def RWR(SM):
    """随机游走 (Random Walk with Restart)"""
    alpha = 0.1
    E = np.identity(len(SM))
    M = np.zeros((len(SM), len(SM)))
    s = []
    # 归一化
    row_sum = np.sum(SM, axis=1)
    row_sum[row_sum == 0] = 1  # 避免除以0
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] = SM[i][j] / row_sum[i]
            
    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10): # 迭代10次
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return np.array(s)

def drug_disease_microbe():
    """
    预处理：利用疾病信息补全关联矩阵
    注意：请确保 ./data/MDAD/ 下有对应文件
    """
    try:
        A = np.loadtxt("./data/MDAD/drug_microbe_matrix.txt")
        dd = np.loadtxt("./data/MDAD/drug_with_disease.txt")
        md = np.loadtxt("./data/MDAD/microbe_with_disease.txt")
        ddict = {}
        mdict = {}
        for i in range(len(dd)):
            ddict[i] = dd[i][1]
        for i in range(len(md)):
            mdict[i] = md[i][1]
        for i in range(1, 110):
            k1 = [k for k, v in ddict.items() if v == i]
            k2 = [k for k, v in mdict.items() if v == i]
            for p in range(len(k1)):
                for q in range(len(k2)):
                    # 注意索引是否需要-1，视数据源而定
                    if int(dd[k1[p]][0])-1 < A.shape[0] and int(md[k2[q]][0])-1 < A.shape[1]:
                        A[int(dd[k1[p]][0]) - 1][int(md[k2[q]][0]) - 1] = 1
        np.savetxt("./data/MDAD/drug_dmicrobe_matrix.txt", A)
        print("Updated matrix saved.")
    except Exception as e:
        print(f"Error in drug_disease_microbe processing: {e}")