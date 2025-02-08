#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import heapq
import sys, re
from sklearn.model_selection import KFold
import os
import xgboost as xgb
from scipy.stats import gmean





def readEmd_program(filename):
    lines = [line.strip("\n\t") for line in open(filename)]
    entity = []
    rep = []
    targetLabel = []
    flag = 0
    for line in lines:
        r = line.split(",")
        targetLabel.append(int(r[0]))
        res = r[1:]
        res_double = [float(val) for val in res]
        rep.append(res_double)
    return rep, targetLabel


_FLAG_TO_DEVICE_NAME = {
    "Cypress": "AMD Radeon HD 5900",
    "Tahiti": "AMD Tahiti 7970",
    "Fermi": "NVIDIA GTX 480",
    "Kepler": "NVIDIA Tesla K20c",
}

device_list = ["Cypress", "Tahiti", "Fermi", "Kepler"]

oracle_file = os.path.join("./data/pact-2014-oracles.csv")
oracles = pd.read_csv(oracle_file)

runtimes_file = os.path.join("./data/pact-2014-runtimes.csv")
df = pd.read_csv(runtimes_file)


ncc_sp_vals = [1.29, 1.07, 0.97, 1.01]
ncc_sp_mean = [1.086]
ir2vec_sp_vals=[1.240354,1.280314,1.229201,1.153582]
ir2vec_sp_mean=[1.225863]

cfs = np.array([1, 2, 4, 8, 16, 32])
kernel_freq = df["kernel"].value_counts().sort_index().reset_index()

def find_runtime(df, kernel, cf, platform):
    filter1 = df["kernel"] == kernel
    filter2 = df["cf"] == cf
    return df.where(filter1 & filter2)["runtime_" + platform].dropna()


def evaluate(max_depth, learning_rate, n_estimators):
    inferencetime = []
    raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 65))
    efileNum = pd.DataFrame(fileIndex)
    embeddings = pd.concat([efileNum, raw_embeddings_pd], axis=1)

    llfiles = pd.read_csv("./data/all.txt", sep="\s+")
    fileNum = llfiles["FileNum"]
    filesname = llfiles["ProgramName"]

    oracles["kernel_path"] = str("./") + oracles["kernel"] + str(".ll")

    df["kernel_path"] = str("./") + df["kernel"] + str(".ll")

    resultant_data = pd.DataFrame()
    for i, platform in enumerate(device_list):
        embeddingsData_tmp = embeddings
        embeddingsData_tmp = embeddingsData_tmp.merge(
            llfiles, left_on=0, right_on="FileNum"
        )
        embeddingsData_tmp = pd.merge(
            embeddingsData_tmp, oracles, left_on="ProgramName", right_on="kernel_path"
        )
        embeddingsData_tmp["cf"] = embeddingsData_tmp["cf_" + platform]
        embeddingsData_tmp["device"] = i + 1
        resultant_data = pd.concat([resultant_data, embeddingsData_tmp])

    resultant_data = pd.get_dummies(resultant_data, columns=["device"])
    resultant_data.reset_index(inplace=True)

    targetLabel = np.array(resultant_data["cf"])
    data = resultant_data
    data = data.drop(
        columns=[
            "index",
            0,
            "FileNum",
            "ProgramName",
            "kernel",
            "cf_Fermi",
            "runtime_Fermi",
            "cf_Kepler",
            "runtime_Kepler",
            "cf_Cypress",
            "runtime_Cypress",
            "cf_Tahiti",
            "runtime_Tahiti",
            "kernel_path",
            "cf",
        ]
    )

    embeddings = (data - data.min()) / (data.max() - data.min())
    embeddings = np.array(embeddings)

    data = []
    kf = KFold(n_splits=len(targetLabel), shuffle=False)
    for j, (train_index, test_index) in enumerate(kf.split(targetLabel)):
        kernel = sorted(set(df["kernel"]))[test_index[0] % 17]
        gbc = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=10,
        )
        gbc.fit(embeddings[train_index], targetLabel[train_index])
        prediction = gbc.predict(embeddings[test_index])[0]

        if embeddings[test_index, 64] == 1:
            platform = device_list[0]
        elif embeddings[test_index, 65] == 1:
            platform = device_list[1]
        elif embeddings[test_index, 66] == 1:
            platform = device_list[2]
        elif embeddings[test_index, 67] == 1:
            platform = device_list[3]

        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        prediction = min(
            prediction, 2 ** (kernel_freq["kernel"][test_index[0] % 17] - 1)
        )
        oracle = targetLabel[test_index[0]]

        rt_baseline = float(find_runtime(df, kernel, 1, platform))
        rt_pred = float(find_runtime(df, kernel, prediction, platform))
        rt_oracle = float(oracle_runtimes[test_index[0] % 17])
        data.append(
            {
                "Model": "binary2vec",
                "Platform": _FLAG_TO_DEVICE_NAME[platform],
                "Kernel": kernel,
                "Oracle-CF": oracle,
                "Predicted-CF": prediction,
                "Speedup": rt_baseline / rt_pred,
                "Oracle": rt_oracle / rt_pred,
                "OracleSpeedUp": rt_baseline / rt_oracle,
            }
        )
    binary2vec = pd.DataFrame(
        data,
        columns=[
            "Model",
            "Platform",
            "Kernel",
            "Oracle-CF",
            "Predicted-CF",
            "Speedup",
            "Oracle",
            "OracleSpeedUp",
        ],
    )

    print("\nSpeedup Matrix: Binary2vec Vs. others\n")
    binary2vec_sp_vals = binary2vec.groupby(["Platform"])["Speedup"].mean().values
    binary2vec_sp_mean = binary2vec_sp_vals.mean()
    sp_df = pd.DataFrame(
        {
            "NCC": ncc_sp_vals + ncc_sp_mean,
            "ir2vec": ir2vec_sp_vals+ir2vec_sp_mean,
            "binary2vec": list(binary2vec_sp_vals) + [binary2vec_sp_mean]
        },
        index=[
            "AMD Radeon HD 5900",
            "AMD Tahiti 7970",
            "NVIDIA GTX 480",
            "NVIDIA Tesla K20c",
            "Average",
        ],
    )
    print(sp_df)



raw_embeddings, fileIndex = readEmd_program('./embeddings/binary2vec_vec.txt')
evaluate(max_depth=10, learning_rate=0.05, n_estimators=40)





