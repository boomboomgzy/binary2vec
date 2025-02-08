import pickle
import numpy as np
import math
import os
from binaryai import BinaryAI
import re
import json
import multiprocessing
import time
import shutil
from scipy.spatial import distance
from tqdm import tqdm
from multiprocessing import Pool
import seaborn 
import matplotlib.pyplot as plt

secret_id=''
secret_key=''

def khash_str_to_list(khash: str) -> list:
    return list(bin(int(khash, 16))[2:].zfill(1024))
 

def khash_similarity(khash_a: str, khash_b: str) -> float:
    khash_a, khash_b = khash_str_to_list(khash_a), khash_str_to_list(khash_b)

    return 2*(1 - distance.hamming(khash_a, khash_b))-1 #将其映射到-1到1


def get_label(filename):
    # 获取倒数第一个 `_` 后的部分
    basename = filename.rsplit('_', 1)[-1]

    result = basename.replace(".txt", "")
    if ".so" in result:
        result = result.split(".so")[0]
    return result

bai = BinaryAI(secret_id=secret_id,secret_key=secret_key) # Initialize the client
from binaryai import BinaryAIFile

file_sha256_mapping = {}
from httpcore import ReadTimeout

label_list=[]





def compute_similarity_row(args):
        i, khash_list, n = args
        row_sim = []
        for j in range(i, n):
            if khash_list[i]=='' or khash_list[j]=='':
                if label_list[i]==label_list[j]:
                    row_sim.append((i,j,1))
                else:
                    row_sim.append((i,j,-1))
            else:                    
                sim = khash_similarity(khash_list[i], khash_list[j])
                row_sim.append((i, j, sim))
        return row_sim

def calculate_similarity_matrix_paralized(khash_list, max_process_num=30):
    n = len(khash_list)
    simi_matrix = np.zeros((n, n))
    
    with tqdm(total=n * (n + 1) // 2, desc="Calculating Similarity Matrix") as pbar:
        with Pool(processes=max_process_num) as pool:
            results = pool.map(compute_similarity_row, [(i, khash_list, n) for i in range(n)])

            # 将计算结果填入相似度矩阵
            for row_sim in results:
                for i, j, sim in row_sim:
                    simi_matrix[i, j] = sim
                    simi_matrix[j, i] = sim
                    pbar.update(1)
    return simi_matrix

def exp_recall_mrr_precision_mse(khash_dir,k):


    recall_1_list=[]
    recall_3_list=[]
    recall_5_list=[]
    #recall_10_list=[]
    mrr_list=[]
    precision_k_list=[]
    khash_list=[]
    global label_list
    mse_list=[]
##    all 
    for khash_file in os.listdir(khash_dir):
        khash_file_path=os.path.join(khash_dir,khash_file)
        with open(khash_file_path,'r') as f:
            khash=f.read()
            if khash=='':
                #continue
                null_count=null_count+1
            khash_list.append(khash)
            label_list.append(get_label(khash_file))
##
    
##   different optimization pair
#    opt_list=['O2','Os']
#    for khash_file in os.listdir(khash_dir):
#        khash_file_path=os.path.join(khash_dir,khash_file)
#        file_name=os.path.basename(khash_file_path)
#        match = re.search(r'_(O[0-3sfast]+)_', file_name)
#        if match:
#            opt_level = match.group(1)
#            if opt_level in opt_list:
#                with open(khash_file_path,'r') as f:
#                    khash=f.read()
#                    khash_list.append(khash)
#                    label_list.append(get_label(khash_file))
#        else:
#            print("optimizetion not found")
#            return 
##   

##  different architectures pair
#    arch_list=['mipseb_32','mips_32']
#    for khash_file in os.listdir(khash_dir):
#        khash_file_path=os.path.join(khash_dir,khash_file)
#        file_name=os.path.basename(khash_file_path)
#        match = re.search(r'_(x86_(32|64)|arm_(32|64)|(mips|mipseb)_32)_', file_name)
#        if match:
#            arch = match.group(1)
#            if arch in arch_list:
#                with open(khash_file_path,'r') as f:
#                    khash=f.read()
#                    khash_list.append(khash)
#                    label_list.append(get_label(khash_file))
#        else:
#            print("arch not found")
#            return 
##

## different compiler pair
#    compiler_list=['clang-9.0','gcc-7.3.0']
#    for khash_file in os.listdir(khash_dir):
#            khash_file_path=os.path.join(khash_dir,khash_file)
#            file_name=os.path.basename(khash_file_path)
#            match = re.search(r'(gcc-\d+\.\d+\.\d+|clang-\d+\.\d+)', file_name)
#            if match:
#                compiler = match.group(1)
#                if compiler in compiler_list:
#                    with open(khash_file_path,'r') as f:
#                        khash=f.read()
#                        khash_list.append(khash)
#                        label_list.append(get_label(khash_file))
#            else:
#                print("compiler not found")
#                return 
##


    #simi_matrix=calculate_similarity_matrix(khash_list)
    simi_matrix=calculate_similarity_matrix_paralized(khash_list)
    for i in range(simi_matrix.shape[0]):
        # 排序每行的余弦相似度，索引越小表示越相似
        similar_indices = np.argsort(simi_matrix[i])[::-1]
        #top_10_simi=similar_indices[0:10]
        top_3_simi=similar_indices[0:3]
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
        if i in top_3_simi:
             recall_3_list.append(1)
        else:
             recall_3_list.append(0)
        #if i in top_10_simi:
        #     recall_10_list.append(1)
        #else:
        #     recall_10_list.append(0)


    recall_1_average = sum(recall_1_list) / len(recall_1_list)
    recall_3_average = sum(recall_3_list) / len(recall_3_list)
    recall_5_average = sum(recall_5_list) / len(recall_5_list)
    #recall_10_average = sum(recall_10_list) / len(recall_10_list)
    avg_mrr=sum(mrr_list) / len(mrr_list)
    avg_precision_k=sum(precision_k_list)/len(precision_k_list)
    avg_mse=sum(mse_list)/len(mse_list)    
    print('avg mrr: ',avg_mrr)
    print(f'avg recall@1: {recall_1_average}')
    print(f'avg recall@3: {recall_3_average}')
    print(f'avg recall@5: {recall_5_average}')
    print(f'avg precision@k: ',avg_precision_k)
    print(f'avg mse: {avg_mse}')




if __name__ == "__main__":
    khash_dir=r''
    exp_recall_mrr_precision_mse(khash_dir,10)
