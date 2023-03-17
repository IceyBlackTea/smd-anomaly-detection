import matplotlib.pyplot as plt


def save_fig(fig_name, file_path, plot_value, predict_result=[], correct_result=[]):
    fig = plt.figure(fig_name, figsize=(200, 4))
    plt.xlabel("time")
    plt.ylabel("value")
    plt.xlim([0, 30000])
    plt.ylim([0, 1])
    plt.ticklabel_format(axis='x', style="plain")

    plt.plot(range(len(plot_value)), plot_value)

    predict_ts = []
    predict_value = []

    correct_ts = []
    correct_value = []

    same_ts = []
    same_value = []

    if len(correct_result) != 0:
        for ts in range(len(predict_result)):
            if predict_result[ts] == 1:
                if correct_result[ts] == 1:
                    same_ts.append(ts)
                    same_value.append(plot_value[ts])
                else:
                    predict_ts.append(ts)
                    predict_value.append(plot_value[ts])

            elif correct_result[ts] == 1:
                correct_ts.append(ts)
                correct_value.append(plot_value[ts])
    else:
         for ts in range(len(predict_result)):
            if predict_result[ts] == 1:
                predict_ts.append(ts)
                predict_value.append(plot_value[ts])

    if len(predict_ts) != 0:
        plt.scatter(predict_ts, predict_value, c='r', s=10)

    if len(correct_ts) != 0:
        plt.scatter(correct_ts, correct_value, c='b', s=10)
    
    if len(same_ts) != 0:
        plt.scatter(same_ts, same_value, c='g', s=10)
    
    fig.savefig(file_path)

    plt.close()


def save_compare_fig(fig_name, file_path, eval_values):
    fig = plt.figure(figsize=(20, 8))

    plt.title(fig_name)
    plt.xlim([0, 39])
    
    plt.xlabel("dimension")

    if fig_name.find("accuracy") != -1:
        plt.ylabel("value")
        plt.ylim([0.5, 1.0])
    elif fig_name.find("f1") != -1:
        plt.ylabel("value")
        plt.ylim([0, 0.5])
    else:
        plt.ylabel("value")
        plt.ylim([0, 1])
    
    legends = []
    for key in eval_values:
        legends.append(key)
        plt.plot(range(1, len(eval_values[key])+1), eval_values[key])
    
    plt.legend(legends, loc="best", shadow=True)

    fig.savefig(file_path)

    plt.close()


def save_time_fig(fig_name, file_path, eval_values):
    fig = plt.figure()

    plt.title(fig_name)
    plt.xlabel("algorithms")
    plt.ylabel("time/s")

    legends = []
    for key in eval_values:
        legends.append(key)

        sum = 0.0
        for value in eval_values[key]:
            sum += value

        plt.bar(key, sum)
    
    plt.legend(legends, loc="best", shadow=True)

    fig.savefig(file_path)

    plt.close()
