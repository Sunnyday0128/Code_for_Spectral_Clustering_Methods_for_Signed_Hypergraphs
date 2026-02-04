import numpy as np
from sklearn.metrics import normalized_mutual_info_score

#计算准确率 acc_cal
def acc_cal(result,real):
    n = len(real)
    real = np.array(real)
    same_num1 = np.sum(result == real)

    real_1 = real.copy()
    real_1 = [1 if x==0 else 0 for x in real_1]
    
    real_1 = np.array(real_1)
    same_num2 = np.sum(result == real_1)
    accuracy = max(same_num1/n,same_num2/n)
    
    # # 临时代码
    # temp_false_num = min(n-same_num1,n-same_num2)
    # print(f'该方法被错误分类的人数为{temp_false_num}，占总人数的{round(temp_false_num/n,2)}')

    return accuracy

#计算归一化互信息NMI
def NMI_cal(result,real):
    nmi = normalized_mutual_info_score(result, real)
    return nmi


# n = 10
# result = [1,0,0,1,1,1,0,0,1,0]
# real = [1,0,0,1,1,1,0,0,1,0]
# a = [0,1,1,0,0,0,1,1,0,1]
# # accuracy = acc_cal(n,result,a)
# # print(accuracy)
# nmi = NMI_cal(result,a)
# print(nmi)