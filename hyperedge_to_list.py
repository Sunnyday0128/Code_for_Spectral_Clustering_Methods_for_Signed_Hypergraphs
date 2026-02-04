import os
import pandas as pd
import numpy as np

def data_pre(file_name):
    # 读取原始CSV文件
    input_file = file_name # 原始超图数据集文件名
    output_file = 'temp.csv'  # 输出文件名

    # 加载数据
    data = pd.read_csv(input_file, header=None)  # 假设没有表头

    # 创建一个空的列表来存储处理后的数据
    formatted_data = []

    # 遍历每一行（每一条超边）
    for edge_index, row in data.iterrows():
        # 获取超边内的节点
        nodes = row.dropna().astype(int).tolist()  # 去掉空值并转换为整数列表
        # 为每个节点添加超边序号
        for node in nodes:
            # 生成标签，80%的概率为1，20%的概率为-1
            label = np.random.choice([1, -1], p=[0.8, 0.2])
            formatted_data.append([node, edge_index, label])

    # 将处理后的数据转换为DataFrame
    formatted_df = pd.DataFrame(formatted_data, columns=['node', 'edge', 'sign'])

    if not os.path.exists(output_file):
        # 创建一个新的空 DataFrame 并保存为 CSV 文件
        empty_df = pd.DataFrame()  # 创建一个空的 DataFrame
        empty_df.to_csv(output_file, index=False, encoding='utf-8')  # 保存为 CSV 文件
    
    # 保存为新的CSV文件
    formatted_df.to_csv(output_file, index=False)

    # print(f"数据已保存为 {output_file}")
    return output_file
