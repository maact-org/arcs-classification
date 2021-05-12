from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import logging
from evaluation.metrics import utils
import numpy as np
import pandas as pd

ACCURACY = "accuracy"
PRECISION = "precision"
RECALL = "recall"
CONFUSION_MATRIX = "confusion_matrix"
CSV_NAME = "metrics.csv"

histogram_metrics = defaultdict(lambda: defaultdict(list))
confussion_matrix = None


def add_single_metric(value, name, tag="tag"):
    """
    Appends a single metric to the set of metrics, appended metrics will be automatically plottable
    :param value: Value of the metric
    :param name: Name of the metric
    """
    histogram_metrics[name][tag].append(value)


def measure_predicted_vs_targets(predicted_list, target_list, item_info=None, batch_size=1, tag="tag"):
    """
    Appends the results of accuracy, precision, recall and confussion matrix to a general set of metrics
    :param predicted_list: Matrix of predicted results of size [batch_size, output_size]
    :param target_list: Matrix of one hot targets of size [batch_size, output_size]
    :param item_info: Information about the items in order to log it
    :param batch_size: Size of the batch
    """
    for predicted, target in list(zip(predicted_list, target_list)):

        predicted_for_cm = predicted.reshape(1, -1)
        target_for_cm = target.toarray().reshape(1, -1)
        current_confussion_matrix = predicted_for_cm * target_for_cm.transpose(1, 0)

        global confussion_matrix

        if confussion_matrix is None:
            confussion_matrix = current_confussion_matrix
        else:
            confussion_matrix += current_confussion_matrix

        metrics = {}
        metrics[ACCURACY] = accuracy_score(predicted, target)
        metrics[PRECISION] = precision_score(predicted, target, average="macro")
        metrics[RECALL] = recall_score(predicted, target, average="macro")

        if metrics[ACCURACY] < 0.5:
            log_message = "found item with a low accuracy of {}: {}"
            logging.info(log_message.format(metrics[ACCURACY], str(item_info)))

        histogram_metrics[ACCURACY][tag].append(metrics[ACCURACY])
        histogram_metrics[PRECISION][tag].append(metrics[PRECISION])
        histogram_metrics[RECALL][tag].append(metrics[RECALL])


def _graph_single_metric(metrics_array, metric_name="", metric_tag="", output_path="", datetime_str=""):
    suffix = ".png"

    # Plots a metric
    plt.plot(metrics_array)
    plt.ylabel(metric_name)

    figure_path = "{}_{}_{}".format(output_path, metric_name, metric_tag)
    figure_path = figure_path + datetime_str + suffix
    plt.savefig(figure_path)
    plt.clf()
    plt.cla()


def graph_metrics(output_path="", labels=[]):
    """
    Plots and saves graphs for all the histogram metrics
    :param output_path: Path for saving the generated images
    """

    datetime_now = datetime.datetime.now()
    datetime_fmt = datetime_now.strftime("%Y%m%d%H%M%S%f")

    for k in histogram_metrics.keys():
        for t in histogram_metrics[k].keys():
            _graph_single_metric(histogram_metrics[k][t],
                                 metric_name=k,
                                 metric_tag=t,
                                 output_path=output_path,
                                 datetime_str=datetime_fmt)

    utils.plot_confusion_matrix(confussion_matrix, labels, normalize=False)


def average_metrics():
    averaged_metrics = defaultdict(lambda: defaultdict(list))
    for k in histogram_metrics.keys():
        for tag in histogram_metrics[k].keys():
            np_a = np.array(histogram_metrics[k][tag])
            averaged_metrics[k][tag] = np.average(np_a)
    return averaged_metrics


def reset_metrics():
    """
    Sets all metrics back to default
    """
    global histogram_metrics
    histogram_metrics = defaultdict(lambda: defaultdict(list))

    global confussion_matrix
    confussion_matrix = None


def save_csv():
    """
    Generates a csv
    """
    tagged_dicts = defaultdict(lambda: defaultdict(list))
    for metric, subdict in dict(histogram_metrics).items():
        for tag, v in subdict.items():
            tagged_dicts[tag][metric].append(v)
    for tag, vals in tagged_dicts.items():
        dict_to_save = {k: v[0] for k, v in vals.items()}
        filename = "{}_{}".format(tag, CSV_NAME)
        df = pd.DataFrame.from_dict(dict_to_save, orient='index')
        df = df.transpose()
        df.to_csv(filename)
