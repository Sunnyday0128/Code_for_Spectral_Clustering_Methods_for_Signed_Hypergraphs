import pandas as pd
import numpy as np

# 数据集来源：https://blog.csdn.net/Moonee_/article/details/132746661

def load_voting_data(file_path):
    """
    加载投票数据集并转换为符号超图关联矩阵
    参数:
        file_path (str): .data文件路径
    返回:
        tuple: (超图关联矩阵, 超边权重矩阵, 节点标签)
    """
    # 读取数据文件
    df = pd.read_csv(file_path, header=None)
    
    # 处理节点标签：默认将共和党标记为1，民主党标记为0

    party_mapping = {'republican': 1, 'democrat': 0}
    node_labels = df.iloc[:, 0].map(party_mapping).values
    
    # 处理投票数据：y→1, n→-1, ?→0
    vote_mapping = {'y': 1, 'n': -1, '?': 0}
    votes = df.iloc[:, 1:].replace(vote_mapping).values
    
    # 生成超图关联矩阵（行：节点，列：超边）
    hypergraph_matrix = votes

    
    return hypergraph_matrix, node_labels

# # 示例用法
# if __name__ == "__main__":
#     file_path = "dataset\\real_dataset\\new_data\\house-votes-84.data"  # 替换为您的文件路径
#     H, W, labels = load_voting_data(file_path)
    
#     print(f"超图关联矩阵维度: {H.shape}")
#     print(f"超边权重矩阵维度: {W.shape}")
#     print(f"节点标签: {labels[:5]}...")