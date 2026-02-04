# 生成具有初始社团结构的符号超图
# 参数：节点数n，超边数m，社团数2，社团大小比例p，超边内节点变异率r
import numpy as np
import pandas as pd

def generate_hyperedges(n, m, p, r):
    # 计算社团一和社团二的节点数目
    num_community_one = int(n * p)
    num_community_two = n - num_community_one

    # 生成节点标号
    nodes = np.arange(1, n + 1)

    # 随机打乱节点顺序
    np.random.shuffle(nodes)

    # 划分社团
    community_one = nodes[:num_community_one].tolist()
    community_two = nodes[num_community_one:].tolist()
    real = np.zeros(n)
    for i in range(n):
        if (i+1) in community_one:
            real[i]=0
        else:
            real[i]=1
    # # 打开文件并写入列表内容
    # with open('real.txt', 'w') as file:
    #     for item in real:
    #         file.write(f"{item}\n")  # 每个元素写入一行

    # print("列表已保存到real.txt 文件中。")     



    # 创建一个空的列表来存储超边数据
    hyperedge_data = []
    
    #按照幂律分布生成超边的大小
    alpha = 2.5
    x_min = 3
    u = np.random.uniform(0, 1, m)
    samples = x_min * (1 - u) ** (-1 / (alpha - 1))
    samples = samples.astype(int)
    # 生成超边
    for hyperedge_index in range(1, m + 1):
        size = samples[hyperedge_index-1]
        if size > n:
            size = n
        # 随机选择节点
        selected_nodes = np.random.choice(nodes, size, replace=False)

        # 为每个节点生成关联值
        for node in selected_nodes:
            if node in community_one:
                # 社团一的节点
                value = 1 if np.random.rand() > r else -1  # 以r的概率变异
            else:
                # 社团二的节点
                value = -1 if np.random.rand() > r else 1  # 以r的概率变异

            # 添加到超边数据列表
            hyperedge_data.append([node, hyperedge_index, value])
    
    # 补齐孤立点，将没有出现在原始数据中的节点补入已有的超边里，这里可能引起幂律分布的改变
    all_nodes = set(range(1,n+1))
    data = np.array(hyperedge_data)
    present_elements = set(data[:, 0])
    missing_elements = all_nodes - present_elements
    for node in missing_elements:
        add_index = np.random.randint(1,m+1)
        r_1 = 0 #默认补入的点都是平衡的
        if node in community_one:
                # 社团一的节点
                value = 1 if np.random.rand() > r_1 else -1  # 以r的概率变异
        else:
                # 社团二的节点
                value = -1 if np.random.rand() > r_1 else 1  # 以r的概率变异
        # 添加到超边数据列表
        hyperedge_data.append([node, add_index, value])
    # i = 0
    # community_one = np.array(community_one)
    # community_two = np.array(community_two)
    
    # for node in missing_elements:
    #     node = node-i
    #     community_one[community_one > node] -= 1
    #     community_two[community_two > node] -= 1
    #     i = i+1
    # community_one = set(community_one)
    # community_two = set(community_two)
    # community_two = community_two - community_one


    # 将数据转换为DataFrame
    hyperedge_df = pd.DataFrame(hyperedge_data, columns=['node', 'edge', 'sign'])

    # 保存为CSV文件
    hyperedge_df.to_csv('hyperedge_data.csv', index=False)
    # print("超边数据已保存为 hyperedge_data.csv")
    #将真实的社团划分情况保存到txt文件中，在此之前先对社团内的元素进行升序排序
    # community_one = np.sort(community_one)
    # community_two = np.sort(community_two)
    # with open('communities.txt', 'w', encoding='utf-8') as file:
    #     file.write(f'社团一为: {community_one}\n')
    #     file.write(f'社团二为: {community_two}\n')
    # print(f'社团一为{community_one}')
    # print(f'社团二为{community_two}')
    
    return real
# # 示例参数
# n = 100    # 总节点数
# m = 600       # 超边数
# p = 0.7      # 社团一的比例
# r = 0.3      # 变异概率

# # 生成超边
# real=generate_hyperedges(n, m, p, r)
# # print(real)
