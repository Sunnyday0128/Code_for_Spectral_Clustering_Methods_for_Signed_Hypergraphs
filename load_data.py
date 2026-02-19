# 读取邻接表数据信息
import pandas as pd
import numpy as np
import hyperedge_to_list
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from filmtrust_pre import filmtrust_pre
from house_vote_pre import load_voting_data

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
    # 计算超边的最小size
    min_hyperedge_size = np.min(hyperedge_sizes)
    # 计算负邻接关系的占比
    total_elements = np.sum(H == -1)+np.sum(H == 1)  # 矩阵中的总元素数量
    negative_count = np.sum(H == -1)  # 负邻接关系的数量
    negative_adj_ratio = negative_count / total_elements
    _log_print('该超图的信息如下：')
    _log_print(f'节点数为{num_nodes}，超边数为{num_edges}')
    _log_print(f'超边的平均size为{round(avg_hyperedge_size,2)}，超边的最大size为{max_hyperedge_size}，超边的最小size为{min_hyperedge_size}，负邻接关系占比为{round(negative_adj_ratio,2)}')
    ave_size = round(avg_hyperedge_size,2)
    
    return num_nodes,num_edges,ave_size,max_hyperedge_size


def edge_size_statisic(H):

    hyperedge_sizes = np.abs(H).sum(axis=0)  # axis=0表示按列求和
    max_size = hyperedge_sizes.max()  # 最大超边大小
    size_counts = np.bincount(hyperedge_sizes)  # 统计每个大小的出现次数
    # 提取有出现的超边大小（从1开始，因为超边至少包含1个节点）
    sizes = np.arange(1, 50)
    counts = size_counts[1:50]  # 排除size=0的情况

    # ----------------------
    # 4. 生成直方图
    plt.figure(figsize=(8, 5))
    plt.bar(sizes, counts, color='#1f77b4', alpha=0.7)
    plt.xlabel('Hyperedge Size', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.title('Distribution of Hyperedge Sizes in Signed Hypergraph', fontsize=20)
    plt.xticks(sizes)  # x轴刻度为实际超边大小
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



    return



def _log_print(content, log_file="real_dataset_attribute.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）

# # 输入数据
# file_name = 'E:\code_for_Spectral_Clustering_Methods_for_Signed_Hypergraphs\hyperedge_data.csv'
# # hyperedge_to_list.data_pre(file_name)
# _log_print(f'当前数据为{file_name}')
# H,A = filmtrust_pre()
# H,real = load_voting_data('dataset\\real_dataset\house-votes-84\house-votes-84.data')
file_name = 'dataset\\real_dataset\pubmed_cocitation.csv'
_log_print(f'当前数据为{file_name}')
file_name = hyperedge_to_list.data_pre(file_name)
H = load_data(file_name)
graph_attribute(H)
# # edge_size_statisic(H)

