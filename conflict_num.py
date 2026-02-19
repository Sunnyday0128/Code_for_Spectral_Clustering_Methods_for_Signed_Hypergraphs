# 计算符号网络上不平衡的pair数目（针对二分类）
import numpy as np
from collections import Counter

def conflict_num(H,partion_1,partion_2):
    normlized_score = 0
    score = 0
    conflict_edge = []
    for i in range(H.shape[1]):
        column = H[:,i]
        # print(column)
        v_pos = [j+1 for j in range(len(column)) if column[j] > 0] #找出在超边中的正节点
        # print(f"v_pos:{v_pos}")
        v_neg = [j+1 for j in range(len(column)) if column[j] < 0] #找出在超边中的负节点
        # print(f"v_neg:{v_neg}")
        if (len(v_pos)+len(v_neg)) != 0: # 这里为了filmtrust数据微调过代码
            normlized_score += 1/(len(v_pos)+len(v_neg))
            if set(v_pos).issubset(set(partion_1)) and set(v_neg).issubset(set(partion_2)):
                score = score
            elif set(v_pos).issubset(set(partion_2)) and set(v_neg).issubset(set(partion_1)):
                score = score
            else:
                score = score + 1/(len(v_pos)+len(v_neg))
                conflict_edge.append(i+1)
    # print(f"冲突边集合：{conflict_edge}")
    normlized_score = score/normlized_score
    return score,normlized_score,conflict_edge

def effect_metric(H,partition_1,partition_2):
    matrix = H
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

    # 首先计算正网络上的情况
    asso_positive = 0
    if len(partition_1) > 1:
        for i in range(len(partition_1)-1):
            index_1 = partition_1[i] - 1
            for j in range(i+1,len(partition_1)):
                index_2 = partition_1[j] - 1
                asso_positive += A_positive[index_1,index_2]
                asso_positive += A_positive[index_2,index_1]
    
    if len(partition_2) > 1:
        for i in range(len(partition_2)-1):
            index_1 = partition_2[i] - 1
            for j in range(i+1,len(partition_2)):
                index_2 = partition_2[j] - 1
                asso_positive += A_positive[index_1,index_2]
                asso_positive += A_positive[index_2,index_1]

    total_asso_positive = np.sum(A_positive) - np.trace(A_positive)
    asso_positive_ratio = asso_positive / total_asso_positive

    # 再计算负网络上的情况
    cut_negative = 0
    if partition_1 and partition_2:
        for node1 in partition_1:
            for node2 in partition_2:
                index_1,index_2 = node1-1,node2-1
                cut_negative += A_negative[index_1,index_2]
                cut_negative += A_negative[index_2,index_1]
    
    total_cut_negative = np.sum(A_negative) - np.trace(A_negative)
    cut_negative_ratio = cut_negative / total_cut_negative


    return asso_positive,asso_positive_ratio,cut_negative,cut_negative_ratio

def _compute_common_elements_sum(A, B):
    # 1. 统计A和B中各元素的出现次数
    count_A = Counter(A)
    count_B = Counter(B)
    
    # 2. 找出两个向量的公共元素
    common_elements = set(A) & set(B)
    
    # 3. 计算乘积之和
    total_sum = 0
    for element in common_elements:
        total_sum += count_A[element] * count_B[element]
    
    return total_sum


def global_new_loss(H,result):
    # 12.22 新loss尝试
    total_score = 0 # 记录整个图上的损失分数
    conflict_edge = [] # 保留参数，无意义，便于代码复用
    m = H.shape[1]
    for i in range(m):
        edge = H[:,i]
        e_score = 0
        v_pos = [j for j in range(len(edge)) if edge[j]>0 ] # 正关联节点
        v_neg = [j for j in range(len(edge)) if edge[j]<0 ] # 负关联节点

        delta_e_plus = len(v_pos) # 正节点数
        delta_e_minus = len(v_neg) # 负节点数

        pos_part = [result[t] for t in v_pos] # 正点所在的cluster编号
        neg_part = [result[t] for t in v_neg] # 负点所在的cluster编号
        
        multi_score = _compute_common_elements_sum(pos_part,neg_part)
        
        if delta_e_plus == 0:
            e_score = (len(set(neg_part))-1)/delta_e_minus
        elif delta_e_minus == 0:
            e_score = (len(set(pos_part))-1)/delta_e_plus
        else:
            e_score = (len(set(pos_part))-1)/delta_e_plus + (len(set(neg_part))-1)/delta_e_minus +  multi_score/(delta_e_plus*delta_e_minus)

        total_score += e_score
        
    return total_score/m,len(conflict_edge)


    
