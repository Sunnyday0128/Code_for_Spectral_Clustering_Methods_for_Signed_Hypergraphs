# 对输入的核矩阵求最小特征值对应的特征向量 并用其做二划分

import numpy as np

def bi_partition(matrix):
    
    #base_line输出结果
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_eigenvalue_index = np.argmin(eigenvalues)
    # if np.min(eigenvalues) == 0:
    #     min_eigenvalue_index += 1
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    # # print(f'当前矩阵的特征向量为{min_eigenvector}')
    # #记录特征向量对应的分类结果
    
    min_eigenvector = min_eigenvector.tolist()

    # print(f'当前矩阵所解得的划分向量为{min_eigenvector}')
    partion_1 = []
    partion_2 = []
    j = 0

    #创建一个数组，将每个节点的社团划分结果记录下来
    result = [0 for _ in range(len(matrix))]
    for i in min_eigenvector:
        if i > 0:
            partion_1.append(j)
            result[j] = 0
        else:
            partion_2.append(j)
            result[j] = 1
        j += 1

    return partion_1,partion_2,result # partition中记录的就是原始节点标号


def bi_partition_2(matrix):

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # 筛选非零特征值及其索引（排除数值接近0的浮点数误差）
    non_zero_mask = np.abs(eigenvalues) > 1e-10  # 1e-10为容差阈值
    non_zero_eigenvalues = eigenvalues[non_zero_mask]
    non_zero_indices = np.where(non_zero_mask)[0]

    if len(non_zero_eigenvalues) == 0:
        raise ValueError("矩阵无非零特征值")

    min_non_zero_index = non_zero_indices[np.argmin(non_zero_eigenvalues)]
    min_non_zero_eigenvector = eigenvectors[:, min_non_zero_index]
    min_eigenvector = min_non_zero_eigenvector
    
    # #base_line输出结果
    # eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # min_eigenvalue_index = np.argmin(eigenvalues)
    # if np.min(eigenvalues) == 0:
    #     min_eigenvalue_index += 1
    # min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    # # print(f'当前矩阵的特征向量为{min_eigenvector}')
    # #记录特征向量对应的分类结果
    
    min_eigenvector = min_eigenvector.tolist()

    # print(f'当前矩阵所解得的划分向量为{min_eigenvector}')
    partion_1 = []
    partion_2 = []
    j = 0

    #创建一个数组，将每个节点的社团划分结果记录下来
    result = []
    for i in min_eigenvector:
        if i > 0:
            partion_1.append(j+1)
            result.append(0)
        else:
            partion_2.append(j+1)
            result.append(1)
        j += 1

    return partion_1,partion_2,result # partition中记录的就是原始节点标号