# 不同形式的kernel矩阵
import numpy as np

def unnorm_signed(matrix:list[list]) -> list[list]:
    H = matrix
    # 分别计算各类所需矩阵
    A = H @ H.T
    A_positive_base = np.zeros_like(A)
    A_positive_base[A>0] = A[A>0]
    A_negative_base = np.zeros_like(A)
    A_negative_base[A<0] = -A[A<0]
    # base_line
    D = np.diag(np.sum(A_positive_base,axis=0)+np.sum(A_negative_base,axis=0))
    # base_L = D-A
    unnorm_base_L = D - A
    return unnorm_base_L

def unnorm_combined(matrix:list[list]) -> list[list]:
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]

    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T
    D_positive = np.diag(np.sum(A_positive,axis=0))
    D_negative = np.diag(np.sum(A_negative,axis=0))
    
    # new_kernel
    # new_K = D_positive-A_positive + A_negative
    unnorm_combined = D_positive-A_positive + A_negative
    return unnorm_combined

def unnorm_spec(matrix:list[list]) -> list[list]:
    H = matrix
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]
    # 分别计算各类所需矩阵
    A = H @ H.T
    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T
    D_positive = np.diag(np.sum(A_positive,axis=0))
    D_negative = np.diag(np.sum(A_negative,axis=0))
    # spectral
    # spec_L = D_positive+D_negative-A
    unnorm_spec_L = D_positive - D_negative - A
    return unnorm_spec_L

def ker(matrix):

    H = matrix
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]
    # 分别计算各类所需矩阵
    A = H @ H.T
    A_positive_base = np.zeros_like(A)
    A_positive_base[A>0] = A[A>0]
    A_negative_base = np.zeros_like(A)
    A_negative_base[A<0] = -A[A<0]

    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T
    D_positive = np.diag(np.sum(A_positive,axis=0))
    D_negative = np.diag(np.sum(A_negative,axis=0))

    # method部分（不同kernel）
    # base_line
    D = np.diag(np.sum(A_positive_base,axis=0)+np.sum(A_negative_base,axis=0))
    # base_L = D-A
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    base_L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    # spectral
    # spec_L = D_positive+D_negative-A
    D_inv_sqrt = np.linalg.inv(np.sqrt(D_positive+D_negative))
    spec_L = np.eye(A.shape[0])-D_inv_sqrt @ A @ D_inv_sqrt
    # new_kernel
    # new_K = D_positive-A_positive + A_negative
    D_inv_sqrt_pos = np.linalg.inv(np.sqrt(D_positive))
    D_inv_sqrt_neg = np.linalg.inv(np.sqrt(D_negative))
    new_K = np.eye(A.shape[0])-D_inv_sqrt_pos @ A_positive @ D_inv_sqrt_pos + D_inv_sqrt_neg @ A_negative @ D_inv_sqrt_neg
    return base_L,spec_L,new_K

def unnorm_adjaceny(matrix:list[list]) -> list[list]:
    H = matrix
    A = H @ H.T
    unnorm_base_L = - A
    # print(unnorm_base_L)
    return unnorm_base_L

def unnorm_adjaceny_zero(matrix:list[list]) -> list[list]:
    H = matrix
    A = H @ H.T
    np.fill_diagonal(A, 0)
    unnorm_base_A = - A
    # print(unnorm_base_L)
    return unnorm_base_A

def alter_combined(matrix,alpha):
    H = matrix
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]
    # 分别计算各类所需矩阵
    A = H @ H.T
    A_positive_base = np.zeros_like(A)
    A_positive_base[A>0] = A[A>0]
    A_negative_base = np.zeros_like(A)
    A_negative_base[A<0] = -A[A<0]

    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T
    D_positive = np.diag(np.sum(A_positive,axis=0))
    D_negative = np.diag(np.sum(A_negative,axis=0))
    
    # new_kernel
    # new_K = D_positive-A_positive + A_negative
    unnorm_combined = alpha*(D_positive-A_positive) + (1-alpha)*A_negative
    return unnorm_combined

def L_plus(matrix):

    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]

    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    D_positive = np.diag(np.sum(A_positive,axis=0))

    L_plus = D_positive-A_positive
    return L_plus

def A_minus(matrix):
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]

    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T
    
    np.fill_diagonal(A_negative, 0)
    A_minus =  A_negative
    return A_minus

def A_plus(matrix):
    
    # 创建正元素矩阵
    positive_H = np.zeros_like(matrix)
    positive_H[matrix > 0] = matrix[matrix > 0]
    # 创建负元素矩阵
    negative_H = np.zeros_like(matrix)
    negative_H[matrix < 0] = -matrix[matrix < 0]

    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    np.fill_diagonal(A_positive,0)
    A_plus = - A_positive # 因为对原问题是求极大，所以如果【应用求最小】加个负号即可
    return A_plus
