import os
import time

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.copod import COPOD

import utils
import visual
import score

def base_iforest(arr):
    clf = IForest(n_estimators=100, max_samples=256, contamination=0.04).fit(arr)
    return clf


def base_hbos(arr):
    clf = HBOS(contamination=0.04).fit(arr)
    return clf


def base_knn(arr):
    clf = KNN(contamination=0.04).fit(arr)
    return clf


def base_lof(arr):
    clf = LOF(contamination=0.04).fit(arr)
    return clf


def base_copod(arr):
    clf = COPOD(contamination=0.04).fit(arr)
    return clf


def detect_train(arr, method):
    clf = None
    if method == "iforest":
        clf = base_iforest(arr)
    elif method == "hbos":
        clf = base_hbos(arr)
    elif method == "knn":
        clf = base_knn(arr)
    elif method == "lof":
        clf = base_lof(arr)
    elif method == "copod":
        clf = base_copod(arr)
    elif method == "iforest_all_dim":
        clf = base_iforest(arr)
    elif method == "copod_all_dim":
        clf = base_copod(arr)


    return clf


def detect(method):
    print("detect method: {}".format(method))

    base_dir = os.path.join(".", "ServerMachineDataset")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    # test_lable_dir = os.path.join(base_dir, "test_label")
    inter_label_dir = os.path.join(base_dir, "interpretation_label")

    output_dir = os.path.join(".", "output")

    # filename = "machine-1-1"

    files = os.listdir(train_dir)

    files.sort()

    evals = {}
    for file in files:
        filename = file[:-4]
        print(filename)
        
        # train
        print("training")
        train_file_path = os.path.join(train_dir, file)
        train_table = utils.read_data_file(train_file_path)
        train_dimension_arrs = utils.split_table(train_table)

        start_time = time.time()
        clf = detect_train(train_table, method)
        end_time = time.time()
        cost_time = end_time - start_time

        train_time = cost_time
        
        # draw train data
        figs_dir = os.path.join(output_dir, method, "figs", filename, "train")
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)

        for i in range(38):
            img_path = os.path.join(figs_dir, str(i + 1) + ".png")

            result = clf.labels_
            visual.save_fig(filename+"train"+str(i), img_path, train_dimension_arrs[i], result)


        # predict
        print("predicting")

        test_file_path = os.path.join(test_dir, file)
        test_table = utils.read_data_file(test_file_path)
        test_dimension_vec_arrs = utils.split_vec_table(test_table)
        test_dimension_arrs = utils.split_table(test_table)
        start_time = time.time()
        predict_result = clf.predict(np.array(test_table))
        end_time = time.time()
        cost_time = end_time - start_time
        
        predict_result = predict_result
        predict_time = cost_time


        # evaluating
        print("evaluating")

        inter_file_path = os.path.join(inter_label_dir, file)
        inter_table, dim_infos = utils.read_interpretation_file(inter_file_path)

        correct_results = []
        for i in range(38):
            correct_result = np.ones(len(predict_result)) * 0
            if i in inter_table:
                for ts in inter_table[i]:
                    correct_result[ts] = 1
            
            correct_results.append(correct_result)

        csv_dir = os.path.join(output_dir, method, "csv")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        for i in range(38):
            values = score.get_score_values(predict_result, correct_results[i])
            accuracy = score.get_accuracy(values)
            precision = score.get_precision(values)
            recall = score.get_recall(values)
            f1 = score.get_f1(values)

            if file not in evals:
                evals[file] = {}
            
            evals[file][i+1] = {
                "dimension": str(i + 1),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_time": train_time,
                "predict_time": predict_time,
            }

        csv_file_path = os.path.join(csv_dir, filename + "_evals.csv")

        score.to_csv(csv_file_path, evals[file], dim_infos)

        # visualize
        print("visualizing")
        figs_dir = os.path.join(output_dir, method, "figs", filename, "test")
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)

        for i in range(38):
            img_path = os.path.join(figs_dir, str(i + 1) + ".png")
            visual.save_fig(filename+"test"+str(i), img_path, test_dimension_arrs[i], predict_result, correct_results[i])
           
    total_dim_aver_values = {}
    targets = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "train_time",
            "predict_time"
        ]
    
    for i in range(38):
        total_dim_aver_values[i+1] = {}
        total_dim_aver_values[i+1]["dimension"] = str(i + 1)
        for target in targets:
            sum = 0.0
            count = 0
            aver = 0.0

            for file in evals:
                if i+1 in evals[file]:
                    dim_value = evals[file][i+1][target]
                    sum += dim_value
                    count += 1

            if count != 0:
                aver = sum / count
            
            total_dim_aver_values[i+1][target] = aver

    total_csv_path = os.path.join(csv_dir, "total_evals.csv")
    score.to_csv(total_csv_path, total_dim_aver_values)


def main():
    # methods = ["iforest", "hbos", "knn", "lof", "copod"]
    methods = ["iforest_all_dim"]
    for method in methods:
        detect(method)


if __name__ == "__main__":
    main()

