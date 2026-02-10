# 对给给定的某个网络和ground truth，得出其聚类结果

import numpy as np
from load_data import load_data
from global_conflict_num import global_conflict_num
from global_conflict_num import global_conflict_num_2
from global_conflict_num import global_new_loss
from global_conflict_num import global_new_loss_2
import laplace
from partition import bi_partition
from collections import defaultdict
from collections import deque
import copy

def _get_most_freq_index(lst):
    # 统计元素频次，返回最多数元素所对应的index列表
    # 目的是从划分指示向量中，获得最大的cluster对应的节点索引
    freq = {}
    for idx, num in enumerate(lst):
        freq[num] = freq.get(num, []) + [idx]  # 存储元素对应的所有索引
    
    # 找到频次最高的元素
    max_count = -1
    result = []
    for num, indices in freq.items():
        if len(indices) > max_count:
            max_count = len(indices)
            result = indices  # 更新为当前最高频次元素的索引
    return result

def _choose_kernel(method,cur_H):
    if method == 'signed':
        matrix = laplace.unnorm_signed(cur_H)
    elif method == 'spec':
        matrix = laplace.unnorm_spec(cur_H)
    elif method == 'adj':
        matrix = laplace.unnorm_adjaceny_zero(cur_H)
    elif method == 'composed':
        matrix = laplace.unnorm_combined(cur_H)
    else:
        print('选择了未知的kernel')
        exit()

    return matrix

def hierachical_cluster(file_name,method) :
    # method表示层次聚类中用的方法名称，可以选择spec,adj,composed和signed
    
    H = load_data(file_name)
    n = len(H)
    m = len(H[0])
    # 计算初始的loss rate
    result = [0 for _ in range(n)] # 初始时把所有点分到一个partition里
    indices = _get_most_freq_index(result)
    last_rate = global_conflict_num(H,result)
    index = 1 # 记录新划分社团的可用标号起始值
    partitioning = True
    while partitioning:
        # 选取当前要划分的社团
        last_result = result
        cur_H = [H[j] for j in indices]
        matrix = _choose_kernel(method,np.array(cur_H))
        partition1,partition2,temp_result = bi_partition(matrix)
        for i in partition1:
            result[indices[i]] = index
        for i in partition2:
            result[indices[i]] = index + 1
        index += 2
        cur_rate = global_conflict_num(H,result)
        if cur_rate >= last_rate:
            result = last_result # 如果聚类变差了，就回到之前的结果
            break
        last_rate = cur_rate
        indices = _get_most_freq_index(result)
    num_category = len(set(result)) # 观测一下分类的数量
    return result,num_category

def hierachical_cluster_plus(H,method) :
    # method表示层次聚类中用的方法名称，可以选择spec,adj,composed和signed
    
    # H = load_data(file_name)
    n = len(H)
    m = len(H[0])
    # 计算初始的loss rate
    result = [0 for _ in range(n)] # 初始时把所有点分到一个partition里
    last_rate,num_unbalanced = global_new_loss(H,result)
    group = defaultdict(list)
    group[0] = [i for i in range(n)] # 初始时，group[0]中包含所有节点
    index = 1
    check_queue = deque([0])
    while check_queue:
        key = check_queue.popleft()
        last_unbalanced = num_unbalanced
        last_result = copy.deepcopy(result)
        last_group = copy.deepcopy(group)
        last_index = index
        last_queue = copy.deepcopy(check_queue)  # 此时被检查元素k已经从check_queue中移除了
        
        indices = group[key] # 是否有可能提取到空列表？
        if len(indices) <= 1:   # 可以设定超参数，规定最小社团节点数比例
            continue

        cur_H = [H[j] for j in indices]  # 考虑那些点所构成的子图
        matrix = _choose_kernel(method,np.array(cur_H))
        partition1,partition2,temp_result = bi_partition(matrix)
        
        if len(partition1) == 0 or len(partition2) == 0: # 如果全划分到某个partition中就没有意义了
            continue
        
        for i in partition1: # partition中记录的是在子图中节点的标签号
            result[indices[i]] = index
            group[index].append(indices[i])
        check_queue.append(index)

        for i in partition2:
            result[indices[i]] = index + 1
            group[index+1].append(indices[i])
        check_queue.append(index+1)
        
        index += 2
        cur_rate,num_unbalanced = global_new_loss(H,result)

        # 分类不佳，回溯分类结果
        if cur_rate >= last_rate:
            result = copy.deepcopy(last_result)
            group = copy.deepcopy(last_group)
            index = last_index
            check_queue = copy.deepcopy(last_queue)
            num_unbalanced = last_unbalanced
            continue
        
        group[key].clear()
        last_rate = cur_rate # last_rate记录当前最好的损失分数
                
    num_category = len(set(result)) # 观测一下分类的数量

    return result,num_category

