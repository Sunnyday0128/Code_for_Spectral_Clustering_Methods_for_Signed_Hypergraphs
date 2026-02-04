import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import load_data #载入数据
import laplace  #laplace矩阵
import partition # 分类方法
import conflict_num
import time
import hyperedge_to_list
import generation
import compare
from collections import defaultdict
from datetime import datetime

def log_print(content, log_file="2-way_alpha_iter.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）


def generated_alpha_iter():
    iteration = 20
    start_time = time.time()
    # 输入数据
    n,m = 1000,2000
    p = 0.5
    r = 0.3
    log_print(f'当前生成模型社团平衡率为{p}，关联变异率为{r}')

    nmi_list = defaultdict(list)

    for t in range(iteration):
        real = generation.generate_hyperedges(n, m, p, r)
        original_name = 'hyperedge_data.csv'
        file_name = original_name
        H = load_data.load_data(file_name)
        elapsed_time = time.time() - start_time

        # 编写关于alpha的循环
        for alpha in np.arange(0.1,0.91,0.1):
            unnorm_combined = laplace.alter_combined(H,alpha)
            partition_1,partition_2,result = partition.bi_partition(unnorm_combined)
            nmi = compare.NMI_cal(result,real)
            nmi_list[alpha].append(nmi)
        

        elapsed_time = time.time() - start_time
        log_print(f"已完成{t+1}次仿真，还有{iteration-t-1}次，已用时{elapsed_time}秒")

    final_score = []
    for key in nmi_list.keys():
        # log_print(f'当前处理的alpha值为{key}')
        cur_list = nmi_list[key]
        ave_score = round(float(np.mean(cur_list)),4)
        final_score.append(ave_score)

    log_print('-----------')
    log_print(f'当前生成模型下，不同alpha值对应的nmi值为{final_score}')
    return

def movie_alpha_iter():
    start_time = time.time()
    # 输入数据
    file_name = 'experiment\真实数据集\\movie_data.csv'
    H = load_data.load_data(file_name)
    # 编写关于alpha的循环
    normalized_score_list = []
    log_print(f'当前实验数据为{file_name}')
    for alpha in np.arange(0.05,0.96,0.05):
        unnorm_combined = laplace.alter_combined(H,alpha)
        partition_1,partition_2,result = partition.bi_partition(unnorm_combined)
        normalized_score,conflict_edge = conflict_num.global_new_loss(H,result)
        normalized_score_list.append(normalized_score)
        elapsed_time = time.time() - start_time
        log_print(f'当前处理的alpha值为{alpha},用时{elapsed_time}秒')
    log_print('最终结果为:')
    log_print(f'{normalized_score_list}')
    return

if __name__ == "__main__":
    current_time = datetime.now()
    log_print('----------------------------------------------------')
    log_print(f'当前时间为{current_time}')
    generated_alpha_iter()
    log_print('----------------------------------------------------')
