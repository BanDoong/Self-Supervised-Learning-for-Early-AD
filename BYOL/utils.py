import os
from shutil import copyfile
import torch
from torchmetrics import ConfusionMatrix
import pandas as pd
from operator import add


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


def to_cpu(inputs):
    return inputs.cpu()


def cf_matrix(y_true, y_pred, num_label):
    cf = ConfusionMatrix(num_classes=num_label)
    _, y_pred = torch.max(y_pred, 1)
    y_pred = to_cpu(y_pred)
    y_true = to_cpu(y_true)
    return cf(y_true, y_pred).flatten()


def to_numpy(acc, spe, sen, f1, prec):
    acc = acc.numpy()
    spe = spe.numpy()
    sen = sen.numpy()
    f1 = f1.numpy()
    prec = prec.numpy()
    return acc, spe, sen, f1, prec


def add_element(cf, new):
    return torch.as_tensor(list(map(add, cf, new)))


def cal_metric(cf_list):
    TN, FP, FN, TP = cf_list
    print(f'TN, FP, FN, TP : {cf_list}')
    accuracy = (TP + TN) / (FP + FN + TP + TN) if FP + FN + TP + TN != 0 else torch.as_tensor(0.)
    # specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else torch.as_tensor(0.)
    precision = TP / (TP + FP) if (TP + FP) != 0 else torch.as_tensor(0.)
    # sensitivity or recall
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else torch.as_tensor(0.)
    bacc = (sensitivity + specificity) / 2
    # F1 = TP / (TP + (FN + FP) / 2) if TP + (FN + FP) / 2 != 0 else torch.as_tensor(0.)
    ppv = TP / (TP + FP) if (TP + FP) != 0 else torch.as_tensor(0.)
    npv = TN / (TN + FN) if (TN + FN) != 0 else torch.as_tensor(0.)

    if (precision + sensitivity) != 0:
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        F1 = torch.as_tensor(0.)
    accuracy, specificity, sensitivity, ppv, npv = to_numpy(accuracy, specificity, sensitivity, ppv, npv)
    bacc = bacc.numpy()
    F1 = F1.numpy()
    return accuracy, bacc, specificity, sensitivity, ppv, npv, F1


def write_result(output_list, result_df):
    result_df.loc[output_list[0]] = output_list
    return result_df


def make_df(finetune=False):
    if finetune:
        result_df = pd.DataFrame(
            {'epoch': [], 'acc': [], 'bacc': [], 'specificity': [], 'sensitivity': [], 'ppv': [], 'npv': [], 'f1': []})
    else:
        result_df = pd.DataFrame({'epoch': [], 'train_loss': []})
    return result_df
