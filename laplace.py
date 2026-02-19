# 不同形式的拉普拉斯矩阵
import numpy as np

def norm_signed(matrix: list[list]) -> list[list]:
    H = np.array(matrix)
    A = H @ H.T

    # 向量化计算正负部分
    A_positive_base = np.where(A > 0, A, 0)
    A_negative_base = np.where(A < 0, -A, 0)

    # 计算对角矩阵 D
    diag_D = np.sum(A_positive_base, axis=0) + np.sum(A_negative_base, axis=0)
    # D_inv_sqrt = np.diag(np.where(diag_D > 1e-10, 1 / np.sqrt(diag_D), 0))
    mask = diag_D > 1e-10
    D_inv_sqrt = np.zeros_like(diag_D)
    D_inv_sqrt[mask] = 1/np.sqrt(diag_D[mask])
    D_inv_sqrt = np.diag(D_inv_sqrt)

    # 归一化拉普拉斯矩阵
    norm_base_L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return norm_base_L

def unnorm_signed(matrix:list[list]) -> list[list]:
    H = np.array(matrix)
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

def norm_combined(matrix: list[list]) -> list[list]:
    # 转换为 NumPy 数组以提高性能
    H = np.array(matrix)

    # 创建正元素矩阵和负元素矩阵（向量化操作）
    positive_H = np.where(H > 0, H, 0)
    negative_H = np.where(H < 0, -H, 0)

    # 计算 A_positive 和 A_negative
    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T

    # 计算 D_positive 和 D_negative 的对角线元素（避免构造完整对角矩阵）
    diag_D_positive = np.sum(A_positive, axis=0)
    diag_D_negative = np.sum(A_negative, axis=0)

    # 避免直接求逆，使用伪逆计算 D_inv_sqrt_pos 和 D_inv_sqrt_neg
    mask_pos = diag_D_positive > 1e-10
    D_inv_sqrt_pos = np.zeros_like(diag_D_positive)
    D_inv_sqrt_pos[mask_pos] = 1 / np.sqrt(diag_D_positive[mask_pos])
    D_inv_sqrt_pos = np.diag(D_inv_sqrt_pos)

    mask_neg = diag_D_negative > 1e-10
    D_inv_sqrt_neg = np.zeros_like(diag_D_negative)
    D_inv_sqrt_neg[mask_neg] = 1 / np.sqrt(diag_D_negative[mask_neg])
    D_inv_sqrt_neg = np.diag(D_inv_sqrt_neg)

    # 计算归一化结果
    norm_combined = (
        np.eye(A_positive.shape[0]) 
        - D_inv_sqrt_pos @ A_positive @ D_inv_sqrt_pos 
        + D_inv_sqrt_neg @ A_negative @ D_inv_sqrt_neg
    )

    return norm_combined

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
    H = np.array(matrix)
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

def norm_spec(matrix: list[list]) -> list[list]:
    # 转换为 NumPy 数组以提高性能
    H = np.array(matrix)

    # 创建正元素矩阵和负元素矩阵（向量化操作）
    positive_H = np.where(H > 0, H, 0)
    negative_H = np.where(H < 0, -H, 0)

    # 计算 A_positive 和 A_negative
    A_positive = positive_H @ positive_H.T + negative_H @ negative_H.T
    A_negative = positive_H @ negative_H.T + negative_H @ positive_H.T

    # 计算 D_positive 和 D_negative 的对角线元素（避免构造完整对角矩阵）
    diag_D_positive = np.sum(A_positive, axis=0)
    diag_D_negative = np.sum(A_negative, axis=0)

    # 避免直接求逆，使用伪逆计算 D_inv_sqrt_pos 和 D_inv_sqrt_neg
    mask_pos = diag_D_positive > 1e-10
    D_inv_sqrt_pos = np.zeros_like(diag_D_positive)
    D_inv_sqrt_pos[mask_pos] = 1 / np.sqrt(diag_D_positive[mask_pos])
    D_inv_sqrt_pos = np.diag(D_inv_sqrt_pos)

    mask_neg = diag_D_negative > 1e-10
    D_inv_sqrt_neg = np.zeros_like(diag_D_negative)
    D_inv_sqrt_neg[mask_neg] = 1 / np.sqrt(diag_D_negative[mask_neg])
    D_inv_sqrt_neg = np.diag(D_inv_sqrt_neg)

    # 计算归一化结果
    I = np.eye(A_positive.shape[0])  # 单位矩阵
    norm_spec_L = (
        I 
        - D_inv_sqrt_pos @ A_positive @ D_inv_sqrt_pos 
        + I 
        - D_inv_sqrt_neg @ A_negative @ D_inv_sqrt_neg
    )

    return norm_spec_L

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
    H = np.array(matrix)
    A = H @ H.T
    np.fill_diagonal(A, 0)
    unnorm_base_A = - A
    # print(unnorm_base_L)
    return unnorm_base_A

def unnorm_spec2(matrix):
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
    unnorm_spec_L = D_positive + D_negative - A
    return unnorm_spec_L

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

