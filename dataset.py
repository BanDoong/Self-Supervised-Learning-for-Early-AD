import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib
from trans import generate_pair


def get_data_pretraining(dataset, adni_merge=False, clinica=False, args=None):
    if adni_merge:
        if dataset == 'train':
            label_path = './training.csv'
        else:
            label_path = './test.csv'
    # elif clinica:
    #     label_path = './clinica_total.csv'
    else:
        if dataset == 'train':
            label_path = '../getlabels/train'
        else:
            label_path = '../getlabels/validation'

    data = pd.read_csv(os.path.join(label_path, 'CN.tsv'), sep='\t')
    if args.ad:
        data = pd.concat([data, pd.read_csv(os.path.join(label_path, 'AD.tsv'), sep='\t')],
                         ignore_index=True)
    if args.mci:
        data = pd.concat([data, pd.read_csv(os.path.join(label_path, 'MCI.tsv'), sep='\t')],
                         ignore_index=True)
    return data


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
        path_all = os.path.join(ses_dir, 't1_linear')
        data_path = path_all + '/' + subj + '_' + s + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
        return data_path

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

    else:
        print('You have to choose one of MRI, Tau, Amyloid')

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


# label_path = os.path.join('../final_labels/getlabels', 'train_splits-5', f'split-{0}')
# data_label = ['CN.tsv', 'AD.tsv', 'MCI.tsv']
#
# label = pd.read_csv(os.path.join(label_path, data_label[0]), sep='\t')
# label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[2]), sep='\t')],
#                   ignore_index=True)
# print(label)
# label = label.iloc[:, [0, 2]]
# print(label)
# print(type(label))
#
#
# def clas_generate_sampler(dataset):
#     from torch.utils.data import sampler
#     # df = dataset.subj_list
#     df = dataset
#     n_labels = 2
#     count = np.zeros(n_labels)
#
#     for idx in df.index:
#         label = df['diagnosis'][idx]
#         if label == 0 or label == 'CN':
#             count[0] += 1
#         elif label == 1 or label == 'MCI':
#             count[1] += 1
#     print(count)
#     weight_per_class = 1 / np.array(count)
#     weights = []
#
#     for idx, label in enumerate(dataset['diagnosis']):
#         if label == 'diagnosis':
#             print(idx)
#             print(label)
#         if label == 0 or label == 'CN':
#             weights += [weight_per_class[0]]
#         elif label == 1 or label == 'MCI':
#             weights += [weight_per_class[1]]
#     # print(weights)
#
#     return sampler.RandomSampler(weights)
#
#
# print(list(clas_generate_sampler(label)))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, fold, args, transformation):
        super(Dataset, self).__init__()

        self.dataset = dataset
        self.fold = fold
        self.model = args.model
        self.finetune = args.finetune
        self.modality = args.modality
        self.args = args
        self.trans_tau = args.trans_tau
        self.transformation = transformation
        self.img_mri = []
        self.img_tau = []
        self.img_amyloid = []

        # if args.finetune:
        self.subj_list = _get_data(self.dataset, args.dir_label, args.ad, args.mci, args.num_label, fold)
        print(
            f'Total Subjects Number for Finetunning : {args.finetune}({self.dataset})  :   {len(self.subj_list["participant_id"])}')
        # else:
        #     self.subj_list = get_data_pretraining(self.dataset, clinica=args.clinica, args=args, fold)
        #     print(f'Total Subjects Number for Pretraining {self.dataset}  :   {len(self.subj_list["participant_id"])}')

        for subj in list(self.subj_list['participant_id']):
            if (args.stable or args.finetune) and not args.clinica:
                subj = subj[8:11] + '_S_' + subj[12:16]
            if args.resize and not args.clinica:
                self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm_resized.nii.gz')
                self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask{args.free}_clipnorm_resized.nii.gz')
                self.amyloid_list = os.path.join(args.dir_data, subj,
                                                 f'{subj}_Amyloid_mask{args.free}_clipnorm_resized.nii.gz')
            elif args.clinica:
                self.mri_list = which_tool('MRI', path=args.dir_data, subj=subj, mask=True, resize=args.resize,
                                           t1_linear=args.t1_linear)
                self.tau_list = which_tool('Tau', path=args.dir_data, subj=subj, mask=True, resize=args.resize)
                self.amyloid_list = which_tool('Amyloid', path=args.dir_data, subj=subj, mask=True, resize=args.resize)
            else:
                self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm.nii.gz')
                self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask{args.free}_clipnorm.nii.gz')
                self.amyloid_list = os.path.join(args.dir_data, subj, f'{subj}_Amyloid_mask{args.free}_clipnorm.nii.gz')

            if 'mri' in args.modality:
                self.img_mri.append(np.expand_dims(nib.load(self.mri_list).get_fdata().astype('float32'), 0))
            if 'tau' in args.modality:
                if args.model == 'resize_vit_base':
                    self.img_tau.append(
                        np.expand_dims(nib.load(self.tau_list).get_fdata().astype('float32')[:, :144, :], 0))
                else:
                    self.img_tau.append(np.expand_dims(nib.load(self.tau_list).get_fdata().astype('float32'), 0))
            if 'amyloid' in args.modality:
                if args.model == 'resize_vit_base':
                    self.img_amyloid.append(
                        np.expand_dims(nib.load(self.amyloid_list).get_fdata().astype('float32')[:, :144, :], 0))
                else:
                    self.img_amyloid.append(
                        np.expand_dims(nib.load(self.amyloid_list).get_fdata().astype('float32'), 0))

        self.label = list(self.subj_list['diagnosis'])
        if len(self.label) == len(self.subj_list['participant_id']):
            print(f"Label and Subjects numbers are same as {len(self.label)}")

    def __getitem__(self, index):
        # print(index)
        # print(type(index))
        img_mri_i, img_mri_j, img_tau_i, img_tau_j, img_amyloid_i, img_amyloid_j = 1, 1, 1, 1, 1, 1
        img_mri = self.img_mri[index] if 'mri' in self.modality else 1
        img_tau = self.img_tau[index] if 'tau' in self.modality else 1
        img_amyloid = self.img_amyloid[index] if 'amyloid' in self.modality else 1

        label = self.label[index]

        if self.trans_tau:
            sample = {'img_mri_i': img_mri, 'img_tau_j': img_tau}
            return sample
        else:
            if self.args.transform and self.dataset == 'train':
                if 'mri' in self.modality:
                    img_mri_i, img_mri_j = generate_pair(img_mri, self.args)
                    img_mri_i = self.transformation(img_mri_i)
                    img_mri_j = self.transformation(img_mri_j)
                if 'tau' in self.modality:
                    img_tau_i, img_tau_j = generate_pair(img_tau, self.args)
                if 'amyloid' in self.modality:
                    img_amyloid_i, img_amyloid_j = generate_pair(img_amyloid, self.args)

            if self.transformation and self.finetune:
                img_mri = self.transformation(img_mri)
                sample = {'img_mri': img_mri, 'img_tau': img_tau, 'img_amyloid': img_amyloid, 'label': label}
            elif self.model == 'autoencoder':
                img_mri = self.transformation(img_mri)
                sample = {'img_mri': img_mri, 'img_tau': img_tau, 'img_amyloid': img_amyloid, 'label': label}
            else:
                sample = {'img_mri_i': img_mri_i, 'img_mri_j': img_mri_j, 'img_tau_i': img_tau_i,
                          'img_tau_j': img_tau_j, 'img_amyloid_i': img_amyloid_i, 'img_amyloid_j': img_amyloid_j}

            return sample

    def __len__(self):
        return len(self.subj_list)


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())
