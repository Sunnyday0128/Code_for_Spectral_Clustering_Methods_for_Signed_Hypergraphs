import time
import numpy as np
from graph_gene import k_way_gene
from graph_gene import k_way_gene_multiedge
from hierachical_cluster import hierachical_cluster
from hierachical_cluster import hierachical_cluster_plus
from sklearn.metrics import normalized_mutual_info_score
from datetime import datetime


def log_print(content, log_file="experiment_hier_gene.txt"):
    """同时打印到控制台和日志文件"""
    print(content)  # 控制台输出
    with open(log_file, "a+", encoding="utf-8") as f:
        f.write(content + "\n")  # 文件写入（自动换行）


def function(r):
    iteration = 20
    start_time = time.time()
    n = 1000     # 总节点数
    m = 2000     # 超边数
    m_times = 4  # 超边数是节点数的倍数
    # r = 0     # 关联变异率
    k = 8       # 社团数量
    print(f'当前实验中负邻接关系的变异率为{r}，当前实验的社团个数为{k}')


    result_nmi = [[] for _ in range(4)]
    num_cate = [[] for _ in range(4)]

    for iter in range(iteration):
        # 生成层次数据集
        # real = k_way_gene(k,n,m,r)
        real = k_way_gene_multiedge(k,n,m_times,r)
        file_name = 'hierachical\multi_communities_graph.csv'
        # result_signed,cate_signed = hierachical_cluster(file_name,'signed')
        # result_spec,cate_spec = hierachical_cluster(file_name,'spec')
        # result_adj,cate_adj = hierachical_cluster(file_name,'adj')
        # result_composed,cate_composed = hierachical_cluster(file_name,'composed')

        result_signed,cate_signed = hierachical_cluster_plus(file_name,'signed')
        result_spec,cate_spec = hierachical_cluster_plus(file_name,'spec')
        result_adj,cate_adj = hierachical_cluster_plus(file_name,'adj')
        result_composed,cate_composed = hierachical_cluster_plus(file_name,'composed')

        result_nmi[0].append(normalized_mutual_info_score(real,result_signed))
        result_nmi[1].append(normalized_mutual_info_score(real,result_spec))
        result_nmi[2].append(normalized_mutual_info_score(real,result_adj))
        result_nmi[3].append(normalized_mutual_info_score(real,result_composed))

        num_cate[0].append(cate_signed)
        num_cate[1].append(cate_spec)
        num_cate[2].append(cate_adj)
        num_cate[3].append(cate_composed)

        elapsed_time = time.time() - start_time
        print(f"已完成{iter+1}次仿真，还有{iteration-iter-1}次，已用时{elapsed_time}秒")

    names = ['signed','spec','adj','composed']
    
    log_print(f'当前实验中负邻接关系的变异率为{r}，当前实验的社团个数为{k}')
    for i in range(len(names)):
        log_print(f'当前方法{names[i]}的nmi均值为{np.mean(result_nmi[i])}')
        log_print(f'当前方法{names[i]}的平均分组数为{np.mean(num_cate[i])}')
    
    return

current_time = datetime.now()
log_print('----------------------------------------------------------------------------------------')
log_print(f'实验开始，实验时间为{current_time}')
log_print('本实验使用global_new_loss作为指标，测试其效果')
for r in np.arange(0,0.51,0.05):
    r = round(r,2)
    function(r)
    log_print('----------------------------------------------------------------------------------------')