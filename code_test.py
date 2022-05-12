from typing import Dict, List

import numpy as np


class MetricModule:
    def __init__(self, metrics, n_classes=2):

        self.n_classes = n_classes

        # Check if wanted metrics are implemented
        list_fn = [
            method_name
            for method_name in dir(MetricModule)
            if callable(getattr(MetricModule, method_name))
        ]
        self.metrics = dict()
        for metric in metrics:
            if f"{metric.lower()}_fn" in list_fn:
                self.metrics[metric] = getattr(MetricModule, f"{metric.lower()}_fn")
            else:
                raise NotImplementedError(
                    f"The metric {metric} is not implemented in the module"
                )

    def apply(self, y, y_pred):
        """
        This is a function to calculate the different metrics based on the list of true label and predicted label
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (Dict[str:float]) metrics results
        """

        if y is not None and y_pred is not None:
            results = dict()
            y = np.array(y)
            y_pred = np.array(y_pred)

            for metric_key, metric_fn in self.metrics.items():
                metric_args = list(metric_fn.__code__.co_varnames)
                if "class_number" in metric_args and self.n_classes > 2:
                    for class_number in range(self.n_classes):
                        results[f"{metric_key}-{class_number}"] = metric_fn(
                            y, y_pred, class_number
                        )
                elif "class_number" in metric_args:
                    results[f"{metric_key}"] = metric_fn(y, y_pred, 0)
                else:
                    results[metric_key] = metric_fn(y, y_pred)
        else:
            results = dict()

        return results

    @staticmethod
    def accuracy_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (float) accuracy
        """
        true = np.sum(y_pred == y)

        return true / len(y)

    @staticmethod
    def sensitivity_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) sensitivity
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_positive + false_negative) != 0:
            return true_positive / (true_positive + false_negative)
        else:
            return 0.0

    @staticmethod
    def specificity_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) specificity
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (false_positive + true_negative) != 0:
            return true_negative / (false_positive + true_negative)
        else:
            return 0.0

    @staticmethod
    def ppv_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) positive predictive value
        """
        true_positive = np.sum((y_pred == class_number) & (y == class_number))
        false_positive = np.sum((y_pred == class_number) & (y != class_number))

        if (true_positive + false_positive) != 0:
            return true_positive / (true_positive + false_positive)
        else:
            return 0.0

    @staticmethod
    def npv_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) negative predictive value
        """
        true_negative = np.sum((y_pred != class_number) & (y != class_number))
        false_negative = np.sum((y_pred != class_number) & (y == class_number))

        if (true_negative + false_negative) != 0:
            return true_negative / (true_negative + false_negative)
        else:
            return 0.0

    @staticmethod
    def ba_fn(y, y_pred, class_number):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
            class_number (int): number of the class studied
        Returns:
            (float) balanced accuracy
        """

        return (
                       MetricModule.sensitivity_fn(y, y_pred, class_number)
                       + MetricModule.specificity_fn(y, y_pred, class_number)
               ) / 2

    @staticmethod
    def confusion_matrix_fn(y, y_pred):
        """
        Args:
            y (List): list of labels
            y_pred (List): list of predictions
        Returns:
            (Dict[str:float]) confusion matrix
        """
        true_positive = np.sum((y_pred == 1) & (y == 1))
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        return {
            "tp": true_positive,
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
        }


from run_finetune import cf_matrix, add_element, cal_metric
import torch

label = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0])
predict = torch.Tensor([[0.9, 0.1], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
cf_ = cf_matrix(label, predict, 2)

label = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0])
predict = torch.Tensor([[0.9, 0.1], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
new_cf_ = cf_matrix(label, predict, 2)
cf_ = add_element(cf_, new_cf_)
accuracy, bacc, specificity, sensitivity, ppv, npv = cal_metric(cf_)
TN, FN, FP, TP = cf_
print(f'TP:{TP}, TN : {TN}, FP: {FP}, FP :{FN}')
print(f'ACC : {accuracy}')
print(f'BACC : {bacc}')
print(f'Specificity : {specificity}')
print(f'Sensitivity : {sensitivity}')
print(f'ppv : {ppv}')
print(f'npv : {npv}')

label = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
predict = torch.Tensor(
    [[0.9, 0.1], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7], [0.9, 0.1],
     [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
_, y_pred = torch.max(predict, -1)
print(MetricModule.confusion_matrix_fn(label.numpy(), y_pred.numpy()))
print(MetricModule.accuracy_fn(label.numpy(), y_pred.numpy()))
print(MetricModule.ba_fn(label.numpy(), y_pred.numpy(), 1))
print(MetricModule.specificity_fn(label.numpy(), y_pred.numpy(), 1))
print(MetricModule.sensitivity_fn(label.numpy(), y_pred.numpy(), 1))
print(MetricModule.ppv_fn(label.numpy(), y_pred.numpy(), 1))
print(MetricModule.npv_fn(label.numpy(), y_pred.numpy(), 1))
from sklearn.metrics import confusion_matrix
label = torch.Tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
predict = torch.Tensor(
    [[0.9, 0.1], [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7], [0.9, 0.1],
     [0.7, 0.3], [0.9, 0.1], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7], [0.7, 0.3], [0.3, 0.7]])
_, y_pred = torch.max(predict, -1)
TN, FN, FP, TP = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
print(f'TP:{TP}, TN : {TN}, FP: {FP}, FP :{FN}')
