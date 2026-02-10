# 读取邻接表数据信息
import pandas as pd
import numpy as np
import hyperedge_to_list
from scipy.sparse import csr_matrix

def load_data(file_name:str)->list[list]:
    
    # 读取 CSV 文件
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    # 获取节点和超边列表
    nodes = list(set(df['node']))
    edges = list(set(df['edge']))

    # 创建点边关联矩阵
    num_nodes = len(nodes)
    num_edges = len(edges)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    edge_to_index = {edge: i for i, edge in enumerate(edges)}

    data = np.array(df['sign'])
    row = np.array([node_to_index[node] for node in df['node']])
    col = np.array([edge_to_index[edge] for edge in df['edge']])

    matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_edges))
    H = np.array(matrix.toarray())
    return H 
    # H是关联矩阵的二维列表

def graph_attribute(H):
    # H是节点的关联矩阵
    num_nodes, num_edges = H.shape
    # 计算每条超边的size（非零元素的数量）
    hyperedge_sizes = np.sum(H != 0, axis=0)
    # 计算超边的平均size
    avg_hyperedge_size = np.mean(hyperedge_sizes)
    # 计算超边的最大size
    max_hyperedge_size = np.max(hyperedge_sizes)
    # 计算负邻接关系的占比
    total_elements = np.sum(H == -1)+np.sum(H == 1)  # 矩阵中的总元素数量
    negative_count = np.sum(H == -1)  # 负邻接关系的数量
    negative_adj_ratio = negative_count / total_elements
    print('该超图的信息如下：')
    print(f'节点数为{num_nodes}，超边数为{num_edges}')
    print(f'超边的平均size为{round(avg_hyperedge_size,2)}，超边的最大size为{max_hyperedge_size}，负邻接关系占比为{round(negative_adj_ratio,2)}')
    ave_size = round(avg_hyperedge_size,2)
    
    return num_nodes,num_edges,ave_size,max_hyperedge_size


# 输入数据
# file_name = 'experiment\\真实数据集\\pubmed_cocitation.csv'
# hyperedge_to_list.data_pre(file_name)
# file_name = 'experiment\\真实数据集\\new_movies.csv'
# print(f'当前数据为{file_name}')
# H = load_data(file_name)
# graph_attribute(H)


