import pandas as pd


def to_csv(file_path, dim_values, relation_table = None):
    if len(dim_values) != 0:
        saves = []
        for dim in dim_values:
            saves.append(dim_values[dim])

        if relation_table:
            targets = dim_values[1].keys()

            append_saves = []
            for dim_str in relation_table:
                append_save = {}
                append_save["dimension"] = dim_str

                for target in targets:
                    if target == "dimension":
                        continue

                    dims = dim_str.split(",")
                    sum = 0.0
                    count = 0
                    aver = 0.0

                    for dim in dims:
                        if int(dim) in dim_values:
                            sum += dim_values[int(dim)][target]
                            count += 1
                    
                    if count != 0:
                        aver = sum / count

                    append_save[target] = aver

                append_saves.append(append_save)
            
            saves.extend(append_saves)
        
        df = pd.DataFrame(saves)
        df.to_csv(file_path, index=False)


def from_csv(file_path):
    df = pd.read_csv(file_path)

    evals = {}

    for row in df.itertuples():
        dimension = int(getattr(row, "dimension"))
        accuracy = getattr(row, "accuracy")
        precision = getattr(row, "precision")
        recall = getattr(row, "recall")
        f1 = getattr(row, "f1")
        train_time = getattr(row, "train_time")
        predict_time = getattr(row, "predict_time")

        if file_path.find("all_dims") != -1:
            train_time /= 38
            predict_time /=38

        evals[dimension] = {
            "dimension": str(dimension),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time": train_time,
            "predict_time": predict_time
        }

    return evals


def get_score_values(prect_arr, label_arr):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i in range(len(prect_arr)):
        if prect_arr[i] == 1:
            if label_arr[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label_arr[i] == 1:
                fn += 1
            else:
                tn += 1

    values = {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn
    }
    
    return values


def get_accuracy(values):
    if (values["tp"] + values["tn"]) == 0:
        return 0.0
    
    return 1.0 * (values["tp"] + values["tn"]) / (values["tp"] + values["tn"] + values["fp"] + values["fn"])


def get_recall(values):
    if values["tp"] == 0:
        return 0.0
    
    return 1.0 * values["tp"]  / (values["tp"] + values["fn"])


def get_precision(values):
    if values["tp"] == 0:
        return 0.0
    
    return 1.0 * values["tp"]  / (values["tp"] + values["fp"])


def get_f1(values):
    p = get_precision(values)
    r = get_recall(values)

    if p == 0.0 or r == 0.0:
        return 0.0
    
    return 2 * p * r / (p + r)
