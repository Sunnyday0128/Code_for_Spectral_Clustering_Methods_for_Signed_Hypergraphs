import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import load_data #载入数据
import laplace  #laplace矩阵
import partition # 分类方法
import conflict_num
import time
import hyperedge_to_list


def log_print(content, log_file="experiment_real_gene_negative_iter.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）

def main_real_iter(r):
    iteration = 20
    original_name = 'dataset\\real_dataset\\pubmed_cocitation.csv'
    log_print(f'当前处理数据为{original_name}')
    start_time = time.time()
    num_method = 4      # 当前实验调用的方法数
    method_conflict_edge = [[] for _ in range(num_method)]
    method_score = [[] for _ in range(num_method)]
    method_asso_positive = [[] for _ in range(num_method)]
    method_asso_positive_ratio = [[] for _ in range(num_method)]
    method_cut_negative = [[] for _ in range(num_method)]
    method_cut_negative_ratio = [[] for _ in range(num_method)]

    for t in range(iteration):
        file_name = hyperedge_to_list.data_pre_r(original_name,r)
        H = load_data.load_data(file_name)
        elapsed_time = time.time() - start_time
        print(f'已完成第{t+1}次模拟中，原始超图的处理，用时{elapsed_time}秒')


        # 调用laplace矩阵，以确定不同的谱方法
        unnorm_signed = laplace.unnorm_signed(H)
        unnorm_spec = laplace.unnorm_spec(H)
        unnorm_combined = laplace.unnorm_combined(H)
        unnorm_adj_zero = laplace.unnorm_adjaceny_zero(H)

        matrices = [unnorm_signed,unnorm_spec,unnorm_adj_zero,unnorm_combined]
        names = ['符号网络退化方法','符号超图拉普拉斯矩阵','置零邻接矩阵','符号超图组合方法']
        elapsed_time = time.time() - start_time
        print(f'已完成第{t+1}次模拟中，所有拉普拉斯矩阵构建，用时{elapsed_time}秒')
        for i in range(len(matrices)):
            matrix = matrices[i]
            partition_1,partition_2,result = partition.bi_partition(matrix)
            score,normalized_score,conflict_edge = conflict_num.conflict_num(H,partition_1,partition_2)
            asso_positive,asso_positive_ratio,cut_negative,cut_negative_ratio = conflict_num.effect_metric(H,partition_1,partition_2)
            loss_score = conflict_num.global_new_loss(H,result)
            method_conflict_edge[i].append(len(conflict_edge))
            method_score[i].append(loss_score)
            method_asso_positive[i].append(asso_positive)
            method_asso_positive_ratio[i].append(asso_positive_ratio)
            method_cut_negative[i].append(cut_negative)
            method_cut_negative_ratio[i].append(cut_negative_ratio)
        elapsed_time = time.time() - start_time
        print(f"已完成{t+1}次仿真，还有{iteration-t-1}次，已用时{elapsed_time}秒")

    for i in range(len(matrices)):
        cur_conflict_edge = method_conflict_edge[i]
        ave_conflict_edge = np.mean(cur_conflict_edge)
        log_print(f'方法【{names[i]}】的不平衡超边数为{round(ave_conflict_edge,2)}')
        cur_score = method_score[i]
        ave_score = np.mean(cur_score)
        log_print(f'方法【{names[i]}】的损失分数为{round(ave_score,4)}')
        cur_asso_positive_ratio = method_asso_positive_ratio[i]
        ave_asso_positive_ratio = np.mean(cur_asso_positive_ratio)
        log_print(f'方法【{names[i]}】的正关系聚集率为{round(ave_asso_positive_ratio,4)}')
        cur_cut_negative_ratio = method_cut_negative_ratio[i]
        ave_cut_negative_ratio = np.mean(cur_cut_negative_ratio)
        log_print(f'方法【{names[i]}】的负关系分离率为{round(ave_cut_negative_ratio,4)}')
        print('------------------------------------')
    
    return

for r in np.arange(0.1,0.51,0.1):
    log_print('----------------------------------------------------------------------')
    log_print(f'当前测试的真实网络生成负邻接比例为{r}')
    main_real_iter(r)
    log_print('----------------------------------------------------------------------')
