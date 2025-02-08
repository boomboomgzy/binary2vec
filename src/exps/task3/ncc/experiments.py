import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn 
import matplotlib.pyplot as plt
import csv
import pickle


def get_label(filename):

    result = filename.replace(".strip.preprocessed_seq.csv", "")

    result = result.rsplit('_', 1)[-1]
    if ".so" in result:
        result = result.split(".so")[0]
    return result

def genvec_inst2ll(csv_file_path):
    emb=None
    emb_file_path=r''
    with open(emb_file_path, 'rb') as file:
        emb = pickle.load(file)

    ll_vec=np.zeros(200)
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if int(row[0])!=8564:
                    ll_vec=ll_vec+emb[int(row[0])]

    return ll_vec

def exp_recall_mrr_precision_mse(testset_dir,k):
    recall_1_list=[]
    #recall_5_list=[]
    #recall_10_list=[]
    mrr_list=[]
    precision_k_list=[]
    vec_list=[]
    label_list=[]
    mse_list=[]
##    all 
    for vec_file in os.listdir(testset_dir):
            vec_file_path=os.path.join(testset_dir,vec_file)
            vec_list.append(genvec_inst2ll(vec_file_path))
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
#                    vec_list.append(genvec_inst2ll(vec_file_path))
#                    label_list.append(get_label(vec_file))
#            else:
#                print("optimizetion not found")
#                return 
##   

##  different architectures pair
#    arch_list=['mipseb_32','mips_32']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'_(x86_(32|64)|arm_(32|64)|(mips|mipseb)_32)_', file_name)
#            if match:
#                arch = match.group(1)
#                if arch in arch_list:
#                    vec_list.append(genvec_inst2ll(vec_file_path))
#                    label_list.append(get_label(vec_file))
#            else:
#                print("arch not found")
#                return 
##

## different compiler pair
#    compiler_list=['clang-9.0','gcc-7.3.0']
#    for vec_file in os.listdir(testset_dir):
#            vec_file_path=os.path.join(testset_dir,vec_file)
#            file_name=os.path.basename(vec_file_path)
#            match = re.search(r'(gcc-\d+\.\d+\.\d+|clang-\d+\.\d+)', file_name)
#            if match:
#                compiler = match.group(1)
#                if compiler in compiler_list:
#                    vec_list.append(genvec_inst2ll(vec_file_path))
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
    #print(f'avg recall@5: {recall_5_average}')
    #print(f'avg recall@10: {recall_10_average}')





if __name__=='__main__':
    vec_dir=r''
    exp_recall_mrr_precision_mse(vec_dir,10)