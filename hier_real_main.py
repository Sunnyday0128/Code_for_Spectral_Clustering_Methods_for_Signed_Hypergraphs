# 该代码用于对真实数据集做层次聚类实验

import time
import hyperedge_to_list
import load_data
import laplace
import partition
import conflict_num
import numpy as np
from global_conflict_num import global_conflict_num
from global_conflict_num import global_conflict_num_2
from global_conflict_num import global_new_loss
from global_conflict_num import global_new_loss_2
from hierachical_cluster import hierachical_cluster_plus
from sklearn.metrics import normalized_mutual_info_score
from datetime import datetime
from filmtrust_pre import filmtrust_pre

def log_print(content, log_file="hier_real.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）


iteration = 1 # 按照lossscore下降原则的话，循环多少次的结果都是一样的
original_name = 'dataset\\real_dataset\\filmtrust\\ratings.txt'
start_time = time.time()
num_method = 4      # 当前实验调用的方法数
method_conflict_edge = [[] for _ in range(num_method)]
method_score = [[] for _ in range(num_method)]

method_conflict_edge_hier = [[] for _ in range(num_method)]
method_score_hier = [[] for _ in range(num_method)]
preseverd_num_groups = [[] for _ in range(num_method)]

# 获取当前本地时间
current_time = datetime.now()
log_print('--------------------------------------------------------------------------------------')
log_print(f'实验时间：{current_time}')  
log_print(f'数据集为{original_name}')
log_print('本次使用了global_loss，对比层次聚类相较于二分类的改进效果')
# preseverd_check_result = [[] for _ in range(num_method)]

for t in range(iteration):
    # file_name = hyperedge_to_list.data_pre(original_name)
    file_name = original_name
    # H = load_data.load_data(file_name)
    H,A_trust = filmtrust_pre()
    elapsed_time = time.time() - start_time
    log_print(f'已完成第{t+1}次模拟中，原始超图的处理，用时{elapsed_time}秒')

    # 调用laplace矩阵，以确定不同的谱方法(二分类)
    unnorm_signed = laplace.unnorm_signed(H)
    unnorm_spec = laplace.unnorm_spec(H)
    unnorm_combined = laplace.unnorm_combined(H)
    unnorm_adj_zero = laplace.unnorm_adjaceny_zero(H)

    matrices = [unnorm_signed,unnorm_spec,unnorm_adj_zero,unnorm_combined]
    names = ['符号网络退化方法','符号超图度矩阵相减方法','置零邻接矩阵','符号超图组合方法']
    elapsed_time = time.time() - start_time
    log_print(f'已完成第{t+1}次模拟中，所有拉普拉斯矩阵构建，用时{elapsed_time}秒')
    for i in range(len(matrices)):
        
        matrix = matrices[i]
        partition_1,partition_2,result = partition.bi_partition(matrix)
        normalized_score,conflict_edge = global_new_loss(H,result)
        # score,normalized_score,conflict_edge = conflict_num.conflict_num(H,partition_1,partition_2)
        # method_conflict_edge[i].append(len(conflict_edge))
        method_conflict_edge[i].append(conflict_edge) # 这个变量实际上没有作用
        method_score[i].append(normalized_score)

        # # 插入检查代码
        # if i == 1:
        #     check_1 = result

    # 调用hier方法
    signed_hier,preseverd_sigend = hierachical_cluster_plus(H,'signed')
    spec_hier,preseverd_spec = hierachical_cluster_plus(H,'spec')
    adj_hier,preseverd_adj = hierachical_cluster_plus(H,'adj')
    composed_hier,preseverd_composed = hierachical_cluster_plus(H,'composed')

    # print(f'在第{t}次实验中，两种方法获得聚类结果的nmi值为{normalized_mutual_info_score(check_1,spec_hier)}')
    # print(f'result(bi) is {result}')
    # print(f'hier(spec) is {spec_hier}')
    
    signed_hier_score,signed_hier_num = global_new_loss(H,signed_hier)
    spec_hier_score,spec_hier_num = global_new_loss(H,spec_hier)
    adj_hier_score,adj_hier_num = global_new_loss(H,adj_hier)
    composed_hier_score,composed_hier_num = global_new_loss(H,composed_hier)
    
    method_conflict_edge_hier[0].append(signed_hier_num)
    method_conflict_edge_hier[1].append(spec_hier_num)
    method_conflict_edge_hier[2].append(adj_hier_num)
    method_conflict_edge_hier[3].append(composed_hier_num)

    method_score_hier[0].append(signed_hier_score)
    method_score_hier[1].append(spec_hier_score)
    method_score_hier[2].append(adj_hier_score)
    method_score_hier[3].append(composed_hier_score)

    preseverd_num_groups[0].append(preseverd_sigend)
    preseverd_num_groups[1].append(preseverd_spec)
    preseverd_num_groups[2].append(preseverd_adj)
    preseverd_num_groups[3].append(preseverd_composed)

    elapsed_time = time.time() - start_time
    log_print(f"已完成{t+1}次仿真，还有{iteration-t-1}次，已用时{elapsed_time}秒")
log_print('\n')
log_print(f'当前处理数据为{original_name}，该图的节点数为{len(H)}，超边数为{len(H[0])}')
for i in range(len(matrices)):
    cur_conflict_edge = method_conflict_edge[i]
    ave_conflict_edge = np.mean(cur_conflict_edge)
    var_conflict_edge = max(abs(max(cur_conflict_edge)-ave_conflict_edge),abs(min(cur_conflict_edge)-ave_conflict_edge))
    # print(f'方法【{names[i]}】的[二划分]不平衡超边数为{round(ave_conflict_edge,2)}+-{round(var_conflict_edge,2)}')
    cur_score = method_score[i]
    ave_score = np.mean(cur_score)
    var_score = max(abs(max(cur_score)-ave_score),abs(min(cur_score)-ave_score))
    log_print(f'方法【{names[i]}】的[二划分]损失分数为{round(ave_score,4)}+-{round(var_score,4)}')
    # print(f'方法【{names[i]}】的[多划分]不平衡超边数为{round(np.mean(method_conflict_edge_hier[i]),2)}')
    log_print(f'方法【{names[i]}】的[多划分]损失分数为{round(np.mean(method_score_hier[i]),4)}')
    log_print(f'方法【{names[i]}】的[多划分]分组数为{round(np.mean(preseverd_num_groups[i]),4)}')

    log_print('-----------------------------------------------------------------------------------')