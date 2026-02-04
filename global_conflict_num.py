# 该代码用于编写针对全局计算冲突边的损失分数

from collections import Counter

def global_conflict_num(H,result):
    total_score = 0
    score = 0
    conflict_edge = []
    
    for i in range(H.shape[1]):
        edge = H[:,i]
        v_pos = [j for j in range(len(edge)) if edge[j]>0 ] # 正关联节点
        v_neg = [j for j in range(len(edge)) if edge[j]<0 ] # 负关联节点
        total_score += 1/(len(v_pos)+len(v_neg))
        
        edge_unbalanced = False
        pos_part = [result[t] for t in v_pos]
        neg_part = [result[t] for t in v_neg]
        if len(set(pos_part)) > 1 or len(set(neg_part)) > 1:
            edge_unbalanced = True 
        
        if set(pos_part) == set(neg_part):
            edge_unbalanced = True

        if edge_unbalanced:
            score += 1/(len(v_pos)+len(v_neg))
            conflict_edge.append(i)

        normalized_score = score/total_score
        
    return normalized_score,len(conflict_edge)



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

def global_conflict_num_2(H,result):
    total_score = 0
    conflict_edge = []
    for i in range(H.shape[1]):
        edge = H[:,i]
        v_pos = [j for j in range(len(edge)) if edge[j]>0 ] # 正关联节点
        v_neg = [j for j in range(len(edge)) if edge[j]<0 ] # 负关联节点
        norm_score = 1/(len(v_pos)+len(v_neg))
        
        edge_score = 0
        pos_part = [result[t] for t in v_pos]
        neg_part = [result[t] for t in v_neg]
        edge_score = len(pos_part)+len(neg_part)+_compute_common_elements_sum(pos_part,neg_part)
        
        total_score += edge_score*norm_score
        
        
    return total_score,len(conflict_edge)


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
        
        if delta_e_minus == 0 and delta_e_plus == 0:
            e_score = 0
            m -= 1
        elif delta_e_plus == 0:
            e_score = (len(set(neg_part))-1)/delta_e_minus
        elif delta_e_minus == 0:
            e_score = (len(set(pos_part))-1)/delta_e_plus
        else:
            e_score = (len(set(pos_part))-1)/delta_e_plus + (len(set(neg_part))-1)/delta_e_minus +  multi_score/(delta_e_plus*delta_e_minus)

        total_score += e_score
        
    return total_score/m,len(conflict_edge)

def global_new_loss_2(H,result):
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
            e_score = 0.5*(len(set(pos_part))-1)/delta_e_plus + 0.5*(len(set(neg_part))-1)/delta_e_minus +  multi_score/(delta_e_plus*delta_e_minus)

        total_score += e_score
        
    return total_score/m,len(conflict_edge)