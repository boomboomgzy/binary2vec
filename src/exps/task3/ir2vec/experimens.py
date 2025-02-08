import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt







def get_label(filename):
    # 获取倒数第一个 `_` 后的部分
    basename = filename.rsplit('_', 1)[-1]

    result = basename.replace(".strip.preprocessed.ll", "")
    if ".so" in result:
        result = result.split(".so")[0]
    return result

def exp_recall_mrr_precision_mse(testset_dir,k):
    recall_1_list=[]
    #recall_5_list=[]
    #recall_10_list=[]
    precision_k_list=[]
    mrr_list=[]
    vec_list=[]
    label_list=[]

    mse_list=[]

##    all 
    for vec_file in os.listdir(testset_dir):
            vec_file_path=os.path.join(testset_dir,vec_file)
            vec = np.loadtxt(vec_file_path)
            vec_list.append(vec)
            label_list.append(get_label(vec_file))
##
    
##   different optimization pair
#    opt_list=['O2','Os']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'_(O[0-3sfast]+)_', file_name)
#            if match:
#                opt_level = match.group(1)
#                if opt_level in opt_list:
#                    vec = np.loadtxt(vec_file_path)
#                    vec_list.append(vec)     
#                    label_list.append(get_label(vec_file))
#            else:
#                print("optimizetion not found")
#                return 
##   

##  different architectures pair
#    arch_list=['mips_32','mipseb_32']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'_(x86_(32|64)|arm_(32|64)|(mips|mipseb)_32)_', file_name)
#            if match:
#                arch = match.group(1)
#                if arch in arch_list:
#                    vec = np.loadtxt(vec_file_path)
#                    vec_list.append(vec)    
#                    label_list.append(get_label(vec_file))
#            else:
#                print("arch not found")
#                return 
##

## different compiler pair
#    compiler_list=['gcc-7.3.0','clang-9.0']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'(gcc-\d+\.\d+\.\d+|clang-\d+\.\d+)', file_name)
#            if match:
#                compiler = match.group(1)
#                if compiler in compiler_list:
#                    vec = np.loadtxt(vec_file_path)
#                    vec_list.append(vec)  
#                    label_list.append(get_label(vec_file))
#            else:
#                print("compiler not found")
#                return 
##

    vec_matrix=np.array(vec_list)
    simi_matrix=cosine_similarity(vec_matrix)
  
    for i in range(simi_matrix.shape[0]):
        # 排序每行的余弦相似度，索引越小表示越相似
        similar_indices = np.argsort(simi_matrix[i])[::-1]
        #top_10_simi=similar_indices[0:10]
        #top_5_simi=similar_indices[0:5]
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

        #if i in top_5_simi:
        #     recall_5_list.append(1)
        #else:
        #     recall_5_list.append(0)
        #if i in top_10_simi:
        #     recall_10_list.append(1)
        #else:
        #     recall_10_list.append(0)


    recall_1_average = sum(recall_1_list) / len(recall_1_list)
    #recall_5_average = sum(recall_5_list) / len(recall_5_list)
    #recall_10_average = sum(recall_10_list) / len(recall_10_list)
    avg_mrr=sum(mrr_list) / len(mrr_list)
    avg_precision_k=sum(precision_k_list)/len(precision_k_list)
    avg_mse=sum(mse_list)/len(mse_list)    

    print('avg mrr: ',avg_mrr)
    print(f'avg recall@1: {recall_1_average}')
    print(f'avg precision@k: ',avg_precision_k)
    print(f'avg mse: {avg_mse}')



if __name__=='__main__':
    result_dir=r''
    exp_recall_mrr_precision_mse(result_dir,10)