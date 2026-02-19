import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from house_vote_pre import load_voting_data

# 设置高分辨率
dpi = 300
# 假设H为(300+ x 16)的关联矩阵，real为(300+ x 1)的党派标签向量
# 生成示例数据（实际使用时替换为真实数据）
np.random.seed(42)
# H = np.random.choice([-1, 0, 1], size=(350, 16), p=[0.3, 0.4, 0.3])
# real = np.random.choice([0, 1], size=350)  # 0=民主党，1=共和党
H,real = load_voting_data('dataset\\real_dataset\house-votes-84\house-votes-84.data')

# 假设原始数据：H(350×16)为投票矩阵，real(350,)为党派向量
top_members = range(20)  # 前50名议员ID
top_6_bills = range(6)      # 前6个法案ID（可替换为特定法案ID）

# 生成边列表：仅保留前50议员对前6法案的有效投票
edges = []
for i in top_members:
    for j in top_6_bills:
        if H[i, j] != 0:  # 跳过未投票的记录
            edges.append((i, f"Bill_{j}", H[i, j]))
# 构建二部图：议员（分区0）、法案（分区1）
G = nx.Graph()
G.add_nodes_from(top_members, bipartite=0)
G.add_nodes_from([f"Bill_{j}" for j in top_6_bills], bipartite=1)
G.add_edges_from([(u, v) for u, v, _ in edges])  # 仅添加筛选后的边

# 优化布局：增大scale减少重叠，手动调整法案位置
pos = nx.bipartite_layout(G, nodes=top_members, scale=4)  # scale越大间距越宽
# 将法案节点集中到中间区域（避免分散）
for bill_node in [f"Bill_{j}" for j in top_6_bills]:
    pos[bill_node] = (0, pos[bill_node][1] * 0.6)  # 横向居中，纵向压缩

plt.figure(figsize=(12, 7))  # 适配前50议员+6法案的布局

# 绘制议员节点（民主党/共和党）
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[i for i in top_members if real[i] == 0],  # 民主党
    node_color="#1f78b4", node_size=100, alpha=0.9, label="Democrat"
)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[i for i in top_members if real[i] == 1],  # 共和党
    node_color="#e41a1c", node_size=100, alpha=0.9, label="Republican"
)

# 绘制法案节点（增大尺寸突出显示）
nx.draw_networkx_nodes(
    G, pos,
    nodelist=[f"Bill_{j}" for j in top_6_bills],
    node_color="#999999", node_size=300, alpha=0.8, label="Bill"
)

# 绘制投票边（支持=蓝色，反对=红色，降低透明度减少混乱）
for u, v, vote in edges:
    edge_color = "#1f78b4" if vote == 1 else "#e41a1c"
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)], edge_color=edge_color, alpha=0.5
    )

# 添加图例与标题
plt.legend(fontsize=14, loc="upper right")
plt.title("The voting situations of the first 20 councilors on 6 bills.", fontsize=18, pad=20)
plt.axis("off")  # 隐藏坐标轴
plt.tight_layout()  # 自动调整布局
plt.savefig('figure_revise/house_vote.pdf', dpi=dpi, format='pdf', bbox_inches='tight')
plt.show()