# 该代码用于生成多层次聚类图
# 返回值：每个点所属社团的指示向量

import numpy as np
import pandas as pd
from collections import deque

def _hyperedge_gene(part1,part2,index_start,m_in_this_gene,r):
    hyperedge_data = []
    #按照幂律分布生成超边的大小
    alpha = 2.5
    x_min = 3
    u = np.random.uniform(0, 1, m_in_this_gene)
    samples = x_min * (1 - u) ** (-1 / (alpha - 1))
    samples = samples.astype(int)

    # 超边生成
    for hyperedge_index in range(index_start,index_start+m_in_this_gene):
        size = samples[hyperedge_index%m_in_this_gene]
        if size > len(part1) + len(part2):
            size = len(part1) + len(part2)
        select_nodes = np.random.choice(part1+part2,size,replace=False)
        
        for node in select_nodes:
            if node in part1:
                value = 1 if np.random.rand() > r else -1
            else:
                value = -1 if np.random.rand() > r else -1

            hyperedge_data.append([node, hyperedge_index, value])
    # 孤立点补齐
    all_nodes = set(part1+part2)
    data = np.array(hyperedge_data)
    present_nodes = set(data[:,0])
    miss_nodes = all_nodes - present_nodes
    for node in miss_nodes:
        add_index = np.random.randint(index_start,index_start+m_in_this_gene)
        if node in part1:
            value = 1
        else:
            value = -1
        hyperedge_data.append([node, add_index, value])

    return hyperedge_data

def k_way_gene(k,n,m,r):
    # 假设：k为2的次幂，且节点恰好可以被平分到所有社团里
    # k：社团数 n：节点数 m:超边数 r:变异率
    real = [0 for _ in range(n)]
    nodes = np.arange(0,n)
    np.random.shuffle(nodes)
    one_part_node_num = n // k
    current_part_node_num = 0
    part_index = 0
    # 对节点进行初始划分
    for i in range(n):
        real[nodes[i]] = part_index
        current_part_node_num += 1
        if current_part_node_num >= one_part_node_num:
            current_part_node_num = 0
            part_index += 1
    # 按组分装
    group = [[] for _ in range(k)]
    for i in range(n):
        group[real[i]].append(i)

    # 构建超边
    hyperedge_data = []
    hyperedge_index = 0
    edge_num_in_one = m//(k-1)
    group_queue = deque(group)
    while len(group_queue) > 1:
        part1 = group_queue.popleft()
        part2 = group_queue.popleft()
        hyperedge_data += _hyperedge_gene(part1,part2,hyperedge_index,edge_num_in_one,r)
        hyperedge_index += edge_num_in_one

    # 导出到表格中
    hyperedge_df = pd.DataFrame(hyperedge_data,columns=['node','edge','sign'])
    # 保存为CSV文件
    hyperedge_df.to_csv('hierachical/multi_communities_graph.csv', index=False)

    return real

def k_way_gene_multiedge(k,n,m_times,r):
    # 假设：k为2的次幂，且节点恰好可以被平分到所有社团里
    # k：社团数 n：节点数 m_times:超边是节点数的次幂 r:变异率
    real = [0 for _ in range(n)]
    nodes = np.arange(0,n)
    np.random.shuffle(nodes)
    one_part_node_num = n // k
    current_part_node_num = 0
    part_index = 0
    # 对节点进行初始划分
    for i in range(n):
        real[nodes[i]] = part_index
        current_part_node_num += 1
        if current_part_node_num >= one_part_node_num:
            current_part_node_num = 0
            part_index += 1
    # 按组分装
    group = [[] for _ in range(k)]
    for i in range(n):
        group[real[i]].append(i)

    # 构建超边
    hyperedge_data = []
    hyperedge_index = 0
    group_queue = deque(group)
    while len(group_queue) > 1:
        part1 = group_queue.popleft()
        part2 = group_queue.popleft()
        edge_num_in_one = m_times * (len(part1) + len(part2))
        hyperedge_data += _hyperedge_gene(part1,part2,hyperedge_index,edge_num_in_one,r)
        hyperedge_index += edge_num_in_one
        new_part = part1 + part2
        group_queue.append(new_part)

    # 导出到表格中
    hyperedge_df = pd.DataFrame(hyperedge_data,columns=['node','edge','sign'])
    # 保存为CSV文件
    hyperedge_df.to_csv('hierachical/multi_communities_graph.csv', index=False)

    return real   


# # 调试区
# def test():    
#     real = k_way_gene_multiedge(4,16,2,0)
#     print(real)
#     # print(hyperedge_data)
#     # print(group)

# test()
