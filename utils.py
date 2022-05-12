import os
import pandas as pd


def write_result(output_list, result_df):
    result_df.loc[output_list[0]] = output_list
    return result_df


def make_df(finetune=False):
    if finetune:
        result_df = pd.DataFrame(
            {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_bacc': [],
             'val_bacc': [], 'train_specificity': [],
             'val_specificity': [], 'train_sensitivity': [], 'val_sensitivity': [], 'train_ppv': [], 'val_ppv': [],
             'train_npv': [], 'val_npv': []})
    else:
        result_df = pd.DataFrame({'epoch': [], 'train_loss': [], 'val_loss': []})
    return result_df
