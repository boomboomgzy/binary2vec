#!/usr/bin/env python
# coding: utf-8

import xgboost as xgb
import pandas as pd
import numpy as np
import sys, re
from sklearn.model_selection import KFold
import os
from scipy.stats.mstats import gmean





def binary2vec_readEmd_program(filename):
    lines = [line.strip("\n\t") for line in open(filename)]
    entity = []
    rep = []
    targetLabel = []
    flag = 0
    for line in lines:
        r=line.split(",")
        targetLabel.append(int(r[0]))
        res = r[1:]
        res_double = [float(val) for val in res]
        rep.append(res_double)
    return rep, targetLabel




static_pred_vals = [58.823529, 56.911765]
static_pred_mean = [57.867647]
static_sp_vals = [1.0, 1.0]
static_sp_mean = [1.0]
ncc_pred_vals = [82.79, 81.76]
ncc_pred_mean = [82.275]
ncc_sp_vals = [3.42, 1.39]
ncc_sp_mean = [2.405]
ir2vec_sp_vals=[3.471963,1.433372]
ir2vec_sp_mean=[2.452667]
ir2vec_pred_vals=[90.284006,87.144993]
ir2vec_pred_mean=[88.714499]





llfiles = pd.read_csv("./data/all.txt", sep="\s+")
fileNum = llfiles["FileNum"]
filesname = llfiles["ProgramName"]

device_dict = {"amd": "AMD Tahiti 7970", "nvidia": "NVIDIA GTX 970"}


def binary2vec_evaluate(max_depth=100, learning_rate=0.5, n_estimators=70, seed=104):
    data = []
    rt_label_dict = {"amd": "runtime_cpu", "nvidia": "runtime_gpu"}

    for i, platform in enumerate(device_dict.keys()):
        platform_name = device_dict[platform]

        df = pd.read_csv("./data/cgo17-{}.csv".format(platform))
        df["bench_data"] = (
            df.loc[df["dataset"] != "default", "benchmark"]
            + str("_")
            + df.loc[df["dataset"] != "default", "dataset"]
        )

        df.loc[df["dataset"] == "default", "bench_data"] = df.loc[
            df["dataset"] == "default", "benchmark"
        ]
        df["bench_data_path"] = str("./") + df["bench_data"] + str(".ll")

        raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 65))
        efileNum = pd.DataFrame(fileIndexNum)
        embeddings = raw_embeddings_pd
        embeddingsData = pd.concat([efileNum, embeddings], axis=1)
        embeddingsData = embeddingsData.merge(llfiles, left_on=0, right_on="FileNum")

        df = pd.merge(
            embeddingsData, df, left_on="ProgramName", right_on="bench_data_path"
        )
        targetLabel = np.array([1 if x == "GPU" else 0 for x in df["oracle"].values])

        embeddings = df.drop(
            columns=[
                "dataset",
                "comp",
                "rational",
                "mem",
                "localmem",
                "coalesced",
                "atomic",
                "runtime_cpu",
                "runtime_gpu",
                0,
                "src",
                "seq",
                "bench_data",
                "bench_data_path",
                "ProgramName",
                "FileNum",
                "Unnamed: 0",
                "benchmark",
                "oracle",
            ]
        )
        embeddings = (embeddings - embeddings.min()) / (
            embeddings.max() - embeddings.min()
        )
        embeddings = np.array(embeddings)

        from sklearn.model_selection import StratifiedKFold

        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for j, (train, test) in enumerate(kf.split(embeddings, targetLabel)):

            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                n_jobs=10,
            )
            model.fit(embeddings[train], targetLabel[train])
            predictions = model.predict(embeddings[test])

            predictions = [
                "CPU" if prediction == 0 else "GPU" for prediction in predictions
            ]
            test_df = df.iloc[test].reset_index()
            assert test_df.shape[0] == len(predictions)
            test_df = pd.concat(
                [test_df, pd.DataFrame(predictions, columns=["predictions"])], axis=1
            )

            rt_label = rt_label_dict[platform]
            for idx, row in test_df.iterrows():
                oracle = row["oracle"]
                pred = row["predictions"]
                rt_baseline = row[rt_label]
                rt_oracle = (
                    row["runtime_cpu"] if oracle == "CPU" else row["runtime_gpu"]
                )
                rt_pred = row["runtime_cpu"] if pred == "CPU" else row["runtime_gpu"]
                data.append(
                    {
                        "Model": "binary2vec",
                        "Platform": platform_name,
                        "Oracle Mapping": oracle,
                        "Predicted Mapping": pred,
                        "Correct?": oracle == pred,
                        "Speedup": rt_baseline / rt_pred,
                        "OracleSpeedUp": rt_baseline / rt_oracle,
                    }
                )
        binary2vec = pd.DataFrame(data, index=range(1, len(data) + 1))

    print("Accuracy Matrix: Binary2vec Vs. others\n")
    binary2vec_pred_vals = binary2vec.groupby(["Platform"])["Correct?"].mean().values * 100
    binary2vec_pred_mean = binary2vec_pred_vals.mean()
    accuracy_df = pd.DataFrame(
        {
            "Static Mapping": static_pred_vals + static_pred_mean,
            "NCC": ncc_pred_vals + ncc_pred_mean,
            "ir2vec":ir2vec_pred_vals+ir2vec_pred_mean,
            "binary2vec": list(binary2vec_pred_vals) + [binary2vec_pred_mean],
        },
        index=["AMD Tahiti 7970", "NVIDIA GTX 970", "Average"],
    )
    print(accuracy_df)

    print("\nSpeedup Matrix: Binary2vec Vs. others\n")
    binary2vec_sp_vals = binary2vec.groupby(["Platform"])["Speedup"].mean().values
    binary2vec_sp_mean = binary2vec_sp_vals.mean()
    sp_df = pd.DataFrame(
        {
            "Static Mapping": static_sp_vals + static_sp_mean,
            "NCC": ncc_sp_vals + ncc_sp_mean,
            "ir2vec":ir2vec_sp_vals+ir2vec_sp_mean,
            "binary2vec": list(binary2vec_sp_vals) + [binary2vec_sp_mean],
        },
        index=["AMD Tahiti 7970", "NVIDIA GTX 970", "Average"],
    )
    print(sp_df)


raw_embeddings, fileIndexNum = binary2vec_readEmd_program('./embeddings/binary2vec_vec.txt')
binary2vec_evaluate(max_depth=100, learning_rate=0.5, n_estimators=70, seed=104)



