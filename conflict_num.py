# 计算符号网络上不平衡的pair数目（针对二分类）
import numpy as np
def conflict_num(H,partion_1,partion_2):
    normlized_score = 0
    score = 0
    conflict_edge = []
    for i in range(H.shape[1]):
        column = H[:,i]
        # print(column)
        v_pos = [j for j in range(len(column)) if column[j] > 0] #找出在超边中的正节点
        # print(f"v_pos:{v_pos}")
        v_neg = [j for j in range(len(column)) if column[j] < 0] #找出在超边中的负节点
        # print(f"v_neg:{v_neg}")
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


    
