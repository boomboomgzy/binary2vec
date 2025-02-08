import pickle
import numpy as np
import math
import os

#def distancePSS16(E1,E2):
#    k2 = min(len(E1[1]),len(E2[1]))
#    return np.linalg.norm(E1[0] - E2[0]) + np.linalg.norm(E1[1][:k2] - E2[1][:k2])

def simPSS16(E1,E2):
    k2 = min(len(E1[1]),len(E2[1]))
    return 1-(np.linalg.norm(E1[0] - E2[0]) + np.linalg.norm(E1[1][:k2] - E2[1][:k2]))/math.sqrt(8)

PSSO_emb_file_path=r''

emb_dict=None
with open(PSSO_emb_file_path, "rb") as f:
    emb_dict=pickle.load(f)

from pathlib import Path
import re
import os
import numpy as np
import seaborn 
import matplotlib.pyplot as plt

def calculate_similarity_matrix(vec_list):
    n = len(vec_list)
    simi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            simi_matrix[i, j] = simPSS16(vec_list[i], vec_list[j])
            simi_matrix[j, i] = simi_matrix[i, j]  

    return simi_matrix


def get_label(filename):
    # 获取倒数第一个 `_` 后的部分
    basename = filename.rsplit('_', 1)[-1]

    result = basename.replace(".pth", "")
    if ".so" in result:
        result = result.split(".so")[0]
    return result

def exp_recall_mrr_precision(k):
    recall_1_list=[]
    recall_5_list=[]
    #recall_10_list=[]
    mrr_list=[]
    precision_k_list=[]
    vec_list=[]
    label_list=[]
    mse_list=[]
##    all 
    for bin_name in emb_dict:
        vec_list.append(emb_dict[bin_name])
        label_list.append(get_label(bin_name))
##
    
##   different optimization pair
#    opt_list=['O2','Os']
#    for bin_name in emb_dict:
#        match = re.search(r'_(O[0-3sfast]+)_', bin_name)
#        if match:
#            opt_level = match.group(1)
#            if opt_level in opt_list:
#                vec_list.append(emb_dict[bin_name])
#                label_list.append(get_label(bin_name))   
##   

##  different architectures pair
#    arch_list=['mipseb_32','mips_32']
#    for bin_name in emb_dict:
#        match = re.search(r'_(x86_(32|64)|arm_(32|64)|(mips|mipseb)_32)_', bin_name)
#        if match:
#            arch = match.group(1)
#            if arch in arch_list:
#                vec_list.append(emb_dict[bin_name])
#                label_list.append(get_label(bin_name))  
##

## different compiler pair
#    compiler_list=['clang-9.0','gcc-7.3.0']
#    for bin_name in emb_dict:
#        match = re.search(r'(gcc-\d+\.\d+\.\d+|clang-\d+\.\d+)', bin_name)
#        if match:
#            compiler = match.group(1)
#            if compiler in compiler_list:
#                vec_list.append(emb_dict[bin_name])
#                label_list.append(get_label(bin_name))  
##

    simi_matrix=calculate_similarity_matrix(vec_list)

    for i in range(simi_matrix.shape[0]):
        # 排序每行的余弦相似度，索引越小表示越相似
        similar_indices = np.argsort(simi_matrix[i])[::-1]
        #top_10_simi=similar_indices[0:10]
        top_5_simi=similar_indices[0:5]
        top_k_simi=similar_indices[0:k]
        top_1_simi=similar_indices[0]
        rank = np.where(similar_indices == i)[0][0]  # 查找 i 的位置
        mrr_list.append(1/(rank + 1))


        if top_1_simi == i:
            recall_1_list.append(1)
        else:
            recall_1_list.append(0)

        precision_k=0
        for ind in top_k_simi:
            if label_list[i]==label_list[ind]:
                precision_k=precision_k+simi_matrix[i][ind]

        precision_k_list.append(precision_k/k)
        mse=0
        for ind in similar_indices:
            if label_list[i]==label_list[ind]:
                mse=mse+(1-simi_matrix[i][ind])**2
            else:
                mse=mse+(-1-simi_matrix[i][ind])**2

        mse_list.append(mse/len(similar_indices))


        if i in top_5_simi:
             recall_5_list.append(1)
        else:
             recall_5_list.append(0)
        #if i in top_10_simi:
        #     recall_10_list.append(1)
        #else:
        #     recall_10_list.append(0)


    recall_1_average = sum(recall_1_list) / len(recall_1_list)
    recall_5_average = sum(recall_5_list) / len(recall_5_list)
    #recall_10_average = sum(recall_10_list) / len(recall_10_list)
    avg_mrr=sum(mrr_list) / len(mrr_list)
    avg_precision_k=sum(precision_k_list)/len(precision_k_list)
    avg_mse=sum(mse_list)/len(mse_list)    
    print('avg mrr: ',avg_mrr)
    print(f'avg recall@1: {recall_1_average}')
    print(f'avg recall@5: {recall_5_average}')
    print(f'avg precision@k: ',avg_precision_k)
    print(f'avg mse: {avg_mse}')
    
    
   



if __name__=='__main__':
    exp_recall_mrr_precision(10)