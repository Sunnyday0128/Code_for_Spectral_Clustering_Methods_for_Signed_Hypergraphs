# 该代码用于对生成图数据，应用不同的kernel，计算其nmi和准确率情况

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import load_data #载入数据
import laplace  #laplace矩阵
import partition # 分类方法
import conflict_num
import compare
import time
import generation


# 设置循环测试次数
ground_truth = True
iteration = 50
start_time = time.time()
n = 1000     # 总节点数
m = 2000     # 超边数
p = 0.5     # 社团一的比例
r = 0.25      # 关联变异率
print(f'当前实验中负邻接关系的变异率为{r}，当前实验的社团比例p为{p}')

result_confedge = [[] for _ in range(6)]
result_nmi = [[] for _ in range(6)]
result_acc =[[] for _ in range(6)]


for iter in range(iteration):
    # 生成超图
    generation.generate_hyperedges(n,m,p,r)
    # 输入数据
    file_name = 'hyperedge_data.csv'
    H = load_data.load_data(file_name)

    if ground_truth:
        real = []
    # 读取文件内容
    with open('real.txt', 'r') as file:
        for line in file:
            # 去掉行末的换行符并将每行转换为整数
            real.append(float(line.strip()))

    # 调用laplace矩阵，以确定不同的谱方法
    unnorm_signed = laplace.unnorm_signed(H)
    unnorm_spec = laplace.unnorm_spec(H)
    unnorm_combined = laplace.unnorm_combined(H)
    L_plus = laplace.L_plus(H)
    A_minus = laplace.A_minus(H)
    unnorm_adj_zero = laplace.unnorm_adjaceny_zero(H)
    # A_plus = laplace.A_plus(H)
    # matrices = [unnorm_signed,unnorm_spec,unnorm_adj_zero,unnorm_combined,L_plus,A_minus]
    matrices = [unnorm_signed,unnorm_spec,unnorm_adj_zero,unnorm_combined,L_plus,A_minus]
    for i in range(len(matrices)):
        matrix = matrices[i]
        # 针对半正定的L_plus，需要慎重处理其特征值求解的的问题
        if i != 4:
            partition_1,partition_2,result = partition.bi_partition(matrix)
        else:
            partition_1,partition_2,result = partition.bi_partition_2(matrix)
        # partition_1,partition_2,result = partition.bi_partition(matrix)
        score,normalized_score,conflict_edge = conflict_num.conflict_num(H,partition_1,partition_2)
        if ground_truth:
            accuracy,nmi = compare.acc_cal(result,real),compare.NMI_cal(result,real)
        result_confedge[i].append(len(conflict_edge))
        result_nmi[i].append(nmi)
        result_acc[i].append(accuracy)


    elapsed_time = time.time() - start_time
    print(f"已完成{iter+1}次仿真，还有{iteration-iter-1}次，已用时{elapsed_time}秒")

# 输出结果
# names = ['unnorm_signed','unnorm_spec','unnorm_combined','norm_signed','norm_spec','norm_combined']
# names = ['unnorm_signed','unnorm_spec','unnorm_adj','unnorm_combined','L_plus','A_minus']
names = ['unnorm_signed','unnorm_spec','unnorm_adj','unnorm_combined','L_plus','A_minus']
print(f'当前实验中负邻接关系的变异率为{r}')
print(f'当前实验的社团比例p为{p}')
for i in range(len(names)):
    # cur_method = names[i]
    # ave_confedge = np.mean(result_confedge[i])
    # var_confedge = max(abs(max(result_confedge[i])-ave_confedge),abs(min(result_confedge[i])-ave_confedge))
    # print(f'方法{cur_method}的平均冲突边数为{round(ave_confedge,2)}+-{round(var_confedge,2)}')

    # ave_nmi = np.mean(result_nmi[i])
    # var_nmi = max(abs(max(result_nmi[i])-ave_nmi),abs(min(result_nmi[i])-ave_nmi))
    # print(f'方法{cur_method}的NMI为{round(ave_nmi,4)}+-{round(var_nmi,4)}')

    # ave_acc = np.mean(result_acc[i])
    # var_acc = max(abs(max(result_acc[i])-ave_acc),abs(min(result_acc[i])-ave_acc))
    # print(f'方法{cur_method}的准确率为{round(ave_acc,4)}+-{round(var_acc,4)}') 
    # print('------------------------------------')
    
    # 输出全部数据，用于画箱线图
    print(f'当前方法为{names[i]}')
    acc_num = [round(float(num),4) for num in result_acc[i]]
    print(f'方法{names[i]}的acc全部值为:\n{acc_num}')
    # nmi_num = [round(float(num),4) for num in result_nmi[i]]
    # print(f'当前方法的acc平均值为{np.mean(result_acc[i])}')
    # print(f'当前方法的nmi平均值为{np.mean(result_nmi[i])}')
    print('--------------------------------------')

print('------------------------------------------------------------------------')

# # 输出图的属性
# num_nodes,num_edges,ave_size,max_hyperedge_size = load_data.graph_attribute(H)