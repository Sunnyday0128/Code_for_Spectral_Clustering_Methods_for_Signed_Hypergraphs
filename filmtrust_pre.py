import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def _preprocess_adjacency_and_incident(A,H):
    """移除孤立节点（度为0的节点）
    该处理是由于部分节点（用户）之间存在信任，这样才能做信任网络聚类
    """
    degree = np.sum(A, axis=1)
    # 保留度>0的节点索引
    valid_nodes = np.where(degree > 0)[0]
    # 返回过滤后的邻接矩阵
    return A[valid_nodes][:, valid_nodes],H[valid_nodes]

def filmtrust_pre():
    # -------------------------- 1. 读取数据（保留之前的修正） --------------------------
    df_hyper = pd.read_csv(
        "dataset\\real_dataset\\filmtrust\\ratings.txt",  
        sep=None,  
        usecols=[0, 1, 2],  
        names=['user_id', 'movie_id', 'rating'],  
        header=None,  
        engine="python"
    ).dropna()  # 读取后删除空值

    df_graph = pd.read_csv(
        "dataset\\real_dataset\\filmtrust\\trust.txt",  
        sep=None,  
        usecols=[0, 1, 2],  
        names=['user1', 'user2', 'trust'],  
        header=None,  
        engine="python"
    ).dropna()  # 读取后删除空值

    # -------------------------- 2. 用户ID映射（强制转int） --------------------------
    all_users = pd.concat([df_hyper['user_id'], df_graph['user1'], df_graph['user2']]).unique()
    user_encoder = LabelEncoder()
    user_encoder.fit(all_users)
    num_users = len(all_users)

    # -------------------------- 3. 评分符号化（不变） --------------------------
    def symbolize(rating):
        if rating >= 4: return 1
        elif rating <= 2: return -1
        else: return 0
    df_hyper['symbol'] = df_hyper['rating'].apply(symbolize)

    # -------------------------- 4. 电影ID映射（强制转int） --------------------------
    movie_encoder = LabelEncoder()
    # 关键：用astype(int)转为Python原生int
    df_hyper['movie_idx'] = movie_encoder.fit_transform(df_hyper['movie_id']).astype(int)  
    num_movies = len(movie_encoder.classes_)

    # -------------------------- 5. 超图矩阵H构建（核心修正） --------------------------
    H = np.zeros((num_users, num_movies), dtype=int)
    for _, row in df_hyper.iterrows():
        # 核心修正1：先转numpy数组→取第一个元素→强制转int
        user_idx = int(user_encoder.transform([row['user_id']])[0])  
        # 核心修正2：从row中取movie_idx时，强制转int（即使LabelEncoder返回numpy.int64）
        movie_idx = int(row['movie_idx'])  
        # 现在两个索引都是Python原生int，彻底解决类型问题
        H[user_idx, movie_idx] = row['symbol']  

    # -------------------------- 6. 信任邻接矩阵A构建（同理修正） --------------------------
    df_graph['trust_symbol'] = df_graph['trust']

    A = np.zeros((num_users, num_users), dtype=int)
    for _, row in df_graph.iterrows():
        # 强制转int，避免numpy类型问题
        u1_idx = int(user_encoder.transform([row['user1']])[0])
        u2_idx = int(user_encoder.transform([row['user2']])[0])
        A[u1_idx, u2_idx] = row['trust_symbol']
        A[u2_idx, u1_idx] = row['trust_symbol']

    A,H = _preprocess_adjacency_and_incident(A,H)

    return H, A

# # 测试代码
# if __name__ == "__main__":
#     H, A = filmtrust_pre()
#     print(f"超图矩阵H形状：{H.shape}（用户数×电影数）")
#     print(f"邻接矩阵A形状：{A.shape}（用户数×用户数）")
