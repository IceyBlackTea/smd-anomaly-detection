import os

import score
import visual


def compare(methods, dir_name):
    targets = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "train_time",
        "predict_time"
    ]

    total_evals = {}

    for target in targets:
        total_evals[target] = {}

    for method in methods:
        csv_path = os.path.join(".", "output", method, "csv", "total_evals.csv")

        evals = score.from_csv(csv_path)
        for target in targets:
            total_evals[target][method] = []
            for dim in evals:
                total_evals[target][method].append(evals[dim][target])

    figs_dir = os.path.join(".", "output", dir_name)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)

    for target in total_evals:
        eval_values = total_evals[target]

        fig_name = "{} of different algorithms".format(target)
        fig_path = os.path.join(figs_dir, "{}.png".format(target))

        if target.find("time") == -1:
            visual.save_compare_fig(fig_name, fig_path, eval_values)
        else:
            visual.save_time_fig(fig_name, fig_path, eval_values)


def main():
    methods = ["iforest", "hbos", "knn", "lof", "copod"]
    dir_name = "compare-1"
    compare(methods, dir_name)
    
    methods = ["iforest", "iforest-filter", "iforest_all_dims", "hbos", "hbos-filter"]
    dir_name = "compare-2"
    compare(methods, dir_name)


main()
