# 程序运行的主函数
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import load_data #载入数据
import laplace  #laplace矩阵
import partition # 分类方法
import conflict_num
import compare
from datetime import datetime
from house_vote_pre import load_voting_data
from ciaodvd_pre import ciaodvd_pre
from scipy.linalg import eigh
from filmtrust_pre import filmtrust_pre

# 计算ground_truth(一般图的normalized cut划分)

def _graph_bipartition(adjacency_matrix):
    """
    无向图二划分算法（返回Python列表）
    :param adjacency_matrix: 无向图邻接矩阵 (n×n 二维数组)
    :return: 节点划分标签列表 (元素为1或0)
    """
    # 1. 计算度矩阵D
    degree = np.sum(adjacency_matrix, axis=1)
    D = np.diag(degree)
    # 2. 构建标准化拉普拉斯矩阵
    sqrt_D = np.diag(1.0 / np.sqrt(degree))  # D^(-1/2)
    L = np.eye(len(degree)) - sqrt_D @ adjacency_matrix @ sqrt_D
    # 3. 计算前两个最小特征值及对应特征向量
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, 1])
    # 4. 取第二小特征值对应的Fiedler向量
    fiedler_vector = eigenvectors[:, 1]
    # 5. 生成1/0标签并转换为Python列表
    labels = np.where(fiedler_vector > 0, 1, 0).tolist()
    
    return labels

def _log_print(content, log_file="2-way_ciaodvd.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）

current_time = datetime.now()
_log_print('-----------------------------------------------------------------')
_log_print(f'实验时间为{current_time}')
# 输入数据
file_name = 'dataset\\real_dataset\\CiaoDVD\\movie-ratings.txt'
# H,real = load_voting_data(file_name)
H,A_trust = ciaodvd_pre()
# H,A_trust = filmtrust_pre()
# print(f'图的关联矩阵为\n{H}')

# 调用laplace矩阵，以确定不同的谱方法
unnorm_signed = laplace.unnorm_signed(H)
unnorm_spec = laplace.unnorm_spec(H)
unnorm_adj = laplace.unnorm_adjaceny_zero(H)
unnorm_combined = laplace.unnorm_combined(H)


matrices = [unnorm_signed,unnorm_spec,unnorm_adj,unnorm_combined]
names = ['SL','SHL','SHA','SHC',]

real = _graph_bipartition(A_trust)

# 调用分类方法,输出结果
_log_print(f'当前处理的数据为{file_name}')
for i in range(len(matrices)):
    _log_print(f'当前所用的划分kernel为{names[i]}')
    matrix = matrices[i]
    partition_1,partition_2,result = partition.bi_partition(matrix)
    # print(f'分类一为：{partition_1}')
    # print(f'分类二为：{partition_2}')
    # score,normalized_score,conflict_edge = conflict_num.conflict_num(H,partition_1,partition_2)
    # loss_score,tem_conflict_edge = conflict_num.global_new_loss(H,result)
    # _log_print(f'该方法的lossscore为{loss_score}，冲突边数是{len(conflict_edge)}')

    # asso_positive,asso_positive_ratio,cut_negative,cut_negative_ratio = conflict_num.effect_metric(H,partition_1,partition_2)
    # _log_print(f'该方法的正关系聚集对数量为{asso_positive}，正关系聚集率为{asso_positive_ratio}')
    # _log_print(f'该方法的负关系分离对数量为{cut_negative}，负关系分离率为{cut_negative_ratio}')    
    accuracy,nmi = compare.acc_cal(result,real),compare.NMI_cal(result,real)
    _log_print(f'该方法的准确率为{accuracy}，归一化互信息值为{nmi}')

_log_print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')