import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from itertools import repeat


# from trans import generate_pair


def get_subdir(path, subj):
    subj_dir = os.path.join(path, subj)
    ses = os.listdir(subj_dir)
    for s in ses:
        if 'ses' in s:
            ses_dir = os.path.join(subj_dir, s)
            return ses_dir, s


def which_tool(tool, path, subj, mask=True, resize=False, t1_linear=False):
    ses_dir, s = get_subdir(path, subj)
    group_name = 'classification'

    if t1_linear:
        if tool == 'MRI':
            path_all = os.path.join(ses_dir, 't1_linear')
            data_path = path_all + '/' + subj + '_' + s + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
        elif tool == 'Tau':
            path_all = os.path.join(ses_dir, 'pet_linear', 'coreg')
            data_path = path_all + '/' + subj + '_Tau.nii.gz'
        elif tool == 'Amyloid':
            path_all = os.path.join(ses_dir, 'pet_linear', 'coreg')
            data_path = path_all + '/' + subj + '_Amyloid.nii.gz'
    else:
        if tool == 'MRI':
            path_all = os.path.join(ses_dir, 't1/spm/segmentation/normalized_space')
            if mask and not resize:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_mask_norm.nii.gz'
            elif resize:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_resized.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_norm.nii.gz'

        elif tool == 'Tau':
            path_all = os.path.join(ses_dir, 'pet/preprocessing/group-' + str(group_name))
            if mask:
                data_path = path_all + '/' + subj + '_adni_tau_coreg_mask_norm.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_adni_tau_coreg_norm.nii.gz'

        elif tool == 'Amyloid':
            path_all = os.path.join(ses_dir, 'pet/preprocessing/group-' + str(group_name))
            if mask:
                data_path = path_all + '/' + subj + '_adni_amyloid_coreg_mask_norm.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_adni_amyloid_coreg_norm.nii.gz'

    return data_path


def _get_data(dataset, dir_label, ad, mci, num_label, fold):
    if dataset == 'train':
        label_path = os.path.join(dir_label, 'train_splits-5', f'split-{fold}')
        data_label = ['CN.tsv', 'AD.tsv', 'MCI.tsv']
    else:
        label_path = os.path.join(dir_label, 'validation_splits-5', f'split-{fold}')
        data_label = ['CN_baseline.tsv', 'AD_baseline.tsv', 'MCI_baseline.tsv']

    label = pd.read_csv(os.path.join(label_path, data_label[0]), sep='\t')
    if ad:
        label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[1]), sep='\t')],
                          ignore_index=True)
    if mci:
        label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[2]), sep='\t')],
                          ignore_index=True)

    label = label.iloc[:, [0, 2]]

    for idx in range(len(label['diagnosis'])):
        if num_label == 3:
            if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'Dementia':
                label['diagnosis'][idx] = 2
            elif label['diagnosis'][idx] == 'MCI':
                label['diagnosis'][idx] = 1
            else:
                label['diagnosis'][idx] = 0
        else:
            # ad/mci vs cn
            if ad and mci:
                if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'MCI':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
            # ad vs cn
            elif ad and not mci:
                if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'Dementia':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
            else:
                # cn vs mci
                if label['diagnosis'][idx] == 'MCI':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
    return label


def roi_which_tool(tool, path, subj, side=None, data_type='linear'):
    ses_dir, s = get_subdir(path, subj)
    if tool == 'MRI':
        path_all = os.path.join(ses_dir, f'roi_based/t1_{data_type}')
        data_path = path_all + '/' + subj + '_mri_roi_based_' + side + '_50.nii.gz'
    elif tool == 'Tau':
        path_all = os.path.join(ses_dir, f'roi_based/pet_{data_type}/tau')
        data_path = path_all + '/' + subj + '_tau_roi_based_' + side + '_50.nii.gz'
    elif tool == 'Amyloid':
        path_all = os.path.join(ses_dir, f'roi_based/pet_{data_type}/amyloid')
        data_path = path_all + '/' + subj + '_amyloid_roi_based_' + side + '_50.nii.gz'
    else:
        print('You have to choose one of MRI, Tau, Amyloid')
    return data_path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, fold, args, transformation):
        super(Dataset, self).__init__()

        self.dataset = dataset
        self.fold = fold
        self.finetune = args.finetune
        self.modality = args.modality
        self.args = args
        self.trans_tau = args.trans_tau
        self.transformation = transformation
        self.img_mri_i = []
        self.img_tau_i = []
        self.img_amyloid_i = []
        self.img_mri_j = []
        self.img_tau_j = []
        self.img_amyloid_j = []

        self.roi = args.roi

        self.subj_list = _get_data(self.dataset, args.dir_label, args.ad, args.mci, args.num_label, fold)
        print(
            f'Total Subjects Number for Finetunning : {args.finetune}({self.dataset})  :   {len(self.subj_list["participant_id"])}')
        # self.subj_list = self.subj_list['diagnosis'].sort_index(ascending=True)
        self.label = []
        ban_list = ['009_S_6212', '012_S_6073', '016_S_6809', '027_S_5079', '027_S_6001', '027_S_6842',
                    '027_S_5109',
                    '027_S_6463', '057_S_6869', '070_S_6911', '029_S_6798', '035_S_4464', '041_S_4510',
                    '099_S_6038', '126_S_0680', '129_S_6784', '023_S_6334', '094_S_6278', '114_S_6524']
        self.num_data = 0
        for subj in list(self.subj_list['participant_id']):
            subj_origin = subj
            subj = subj[8:11] + '_S_' + subj[12:16]
            if subj in ban_list:
                pass
            else:
                if self.dataset == 'train':
                    # for aug in range(5):
                    #     self.mri_list_i = os.path.join(args.dir_data, subj,
                    #                                    f'{subj}_MRI_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     self.tau_list_i = os.path.join(args.dir_data, subj,
                    #                                    f'{subj}_Tau_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     self.amyloid_list_i = os.path.join(args.dir_data, subj,
                    #                                        f'{subj}_Amyloid_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     if 'mri' in args.modality:
                    #         self.img_mri_i.append(
                    #             np.expand_dims(nib.load(self.mri_list_i).get_fdata().astype('float32'), 0))
                    #     if 'tau' in args.modality:
                    #         self.img_tau_i.append(
                    #             np.expand_dims(nib.load(self.tau_list_i).get_fdata().astype('float32'), 0))
                    #     if 'amyloid' in args.modality:
                    #         self.img_amyloid_i.append(
                    #             np.expand_dims(nib.load(self.amyloid_list_i).get_fdata().astype('float32'), 0))
                    # for aug in range(5, 10):
                    #     self.mri_list_j = os.path.join(args.dir_data, subj,
                    #                                    f'{subj}_MRI_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     self.tau_list_j = os.path.join(args.dir_data, subj,
                    #                                    f'{subj}_Tau_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     self.amyloid_list_j = os.path.join(args.dir_data, subj,
                    #                                        f'{subj}_Amyloid_mask_norm_crop_resize_aug_{aug}.nii.gz')
                    #     if 'mri' in args.modality:
                    #         self.img_mri_j.append(
                    #             np.expand_dims(nib.load(self.mri_list_j).get_fdata().astype('float32'), 0))
                    #     if 'tau' in args.modality:
                    #         self.img_tau_j.append(
                    #             np.expand_dims(nib.load(self.tau_list_j).get_fdata().astype('float32'), 0))
                    #     if 'amyloid' in args.modality:
                    #         self.img_amyloid_j.append(
                    #             np.expand_dims(nib.load(self.amyloid_list_j).get_fdata().astype('float32'), 0))

                    self.mri_list_i = os.path.join(args.dir_data, subj,
                                                   f'{subj}_MRI_mask_norm_crop_resize_aug_0.nii.gz')
                    self.tau_list_i = os.path.join(args.dir_data, subj,
                                                   f'{subj}_Tau_mask_norm_crop_resize_aug_0.nii.gz')
                    self.amyloid_list_i = os.path.join(args.dir_data, subj,
                                                       f'{subj}_Amyloid_mask_norm_crop_resize_aug_0.nii.gz')
                    self.mri_list_j = os.path.join(args.dir_data, subj,
                                                   f'{subj}_MRI_mask_norm_crop_resize_aug_1.nii.gz')
                    self.tau_list_j = os.path.join(args.dir_data, subj,
                                                   f'{subj}_Tau_mask_norm_crop_resize_aug_1.nii.gz')
                    self.amyloid_list_j = os.path.join(args.dir_data, subj,
                                                       f'{subj}_Amyloid_mask_norm_crop_resize_aug_1.nii.gz')
                    if 'mri' in args.modality:
                        self.img_mri_i.append(
                            np.expand_dims(nib.load(self.mri_list_i).get_fdata().astype('float32'), 0))
                        self.img_mri_j.append(
                            np.expand_dims(nib.load(self.mri_list_j).get_fdata().astype('float32'), 0))
                    if 'tau' in args.modality:
                        self.img_tau_i.append(
                            np.expand_dims(nib.load(self.tau_list_i).get_fdata().astype('float32'), 0))
                        self.img_tau_j.append(
                            np.expand_dims(nib.load(self.tau_list_j).get_fdata().astype('float32'), 0))
                    if 'amyloid' in args.modality:
                        self.img_amyloid_i.append(
                            np.expand_dims(nib.load(self.amyloid_list_i).get_fdata().astype('float32'), 0))
                        self.img_amyloid_j.append(
                            np.expand_dims(nib.load(self.amyloid_list_j).get_fdata().astype('float32'), 0))

                    each_subj = int(self.subj_list[self.subj_list['participant_id'] == subj_origin]['diagnosis'])
                    # self.label += list(repeat(each_subj, 5))
                    self.label += list(repeat(each_subj, 2))
                    # self.num_data += 5
                    self.num_data += 1
                else:
                    self.mri_list = os.path.join(args.dir_data, subj,
                                                 f'{subj}_MRI_mask_norm_crop_resize_aug_0.nii.gz')
                    self.tau_list = os.path.join(args.dir_data, subj,
                                                 f'{subj}_Tau_mask_norm_crop_resize_aug_0.nii.gz')
                    self.amyloid_list = os.path.join(args.dir_data, subj,
                                                     f'{subj}_Amyloid_mask_norm_crop_resize_aug_0.nii.gz')
                    if 'mri' in args.modality:
                        self.img_mri.append(np.expand_dims(nib.load(self.mri_list).get_fdata().astype('float32'), 0))
                    if 'tau' in args.modality:
                        self.img_tau.append(np.expand_dims(nib.load(self.tau_list).get_fdata().astype('float32'), 0))
                    if 'amyloid' in args.modality:
                        self.img_amyloid.append(
                            np.expand_dims(nib.load(self.amyloid_list).get_fdata().astype('float32'), 0))
                    self.label += [int(self.subj_list[self.subj_list['participant_id'] == subj_origin]['diagnosis'])]
                    self.num_data += 1
        print(f'number of labels : {len(self.label)}')
        if len(self.label) == self.num_data and self.dataset == 'train':
            print(f"Pretraining Label and Subjects numbers are same as {self.num_data}")
        elif len(self.label) == self.num_data and self.dataset != 'train':
            print(f"Finetunning Label and Subjects numbers are same as {self.num_data}")

    def __getitem__(self, index):
        # print(index)
        # print(type(index))
        if self.dataset == 'train':
            img_mri_i = self.img_mri_i[index] if 'mri' in self.modality else 1
            img_tau_i = self.img_tau_i[index] if 'tau' in self.modality else 1
            img_amyloid_i = self.img_amyloid_i[index] if 'amyloid' in self.modality else 1
            img_mri_j = self.img_mri_j[index] if 'mri' in self.modality else 1
            img_tau_j = self.img_tau_j[index] if 'tau' in self.modality else 1
            img_amyloid_j = self.img_amyloid_j[index] if 'amyloid' in self.modality else 1
            label = self.label[index]
            sample = {'img_mri_i': img_mri_i, 'img_mri_j': img_mri_j, 'img_tau_i': img_tau_i, 'img_tau_j': img_tau_j,
                      'img_amyloid_i': img_amyloid_i, 'img_amyloid_j': img_amyloid_j, 'label': label}
        else:
            img_mri = self.img_mri[index] if 'mri' in self.modality else 1
            img_tau = self.img_tau[index] if 'tau' in self.modality else 1
            img_amyloid = self.img_amyloid[index] if 'amyloid' in self.modality else 1
            label = self.label[index]
            sample = {'img_mri': img_mri, 'img_tau': img_tau, 'img_amyloid': img_amyloid, 'label': label}
        return sample

    def __len__(self):
        # if self.roi:
        #     return len(self.subj_list) * 2
        # else:
        return self.num_data


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())
