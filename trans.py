from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
import torch
from scipy.special import comb

from preprocessing.genesis import data_augmentation, local_pixel_shuffling, nonlinear_transformation, image_in_painting, \
    image_out_painting



# from preprocessing.cpc import


def generate_pair(img, args, status="test"):
    img_rows, img_cols, img_deps = img.shape[1], img.shape[2], img.shape[3]
    # while True:
    # index = [i for i in range(img.shape[0])]
    # random.shuffle(index)
    # y = img[index[:args.batch_size]]
    y = img
    # transform = x & original = y
    x = copy.deepcopy(y)
    # x = copy.deepcopy(y)
    # for n in range(args.batch_size):
    # Autoencoder

    # Flip
    if args.flip:
        x, y = data_augmentation(x, y, args.flip_rate)

    # Local Shuffle Pixel
    if args.local_shuffle_pixel:
        x = local_pixel_shuffling(x, prob=args.local_rate)
        y = local_pixel_shuffling(y, prob=args.local_rate)

    # Apply non-Linear transformation with an assigned probability
    if args.non_inear_ransformation:
        x = nonlinear_transformation(x, args.nonlinear_rate)
        y = nonlinear_transformation(y, args.nonlinear_rate)

    # Inpainting
    if args.in_painting:
        if random.random() < args.paint_rate:
            if random.random() < args.inpaint_rate:
                x = image_in_painting(x)
                y = image_in_painting(y)

    # Outpainting
    if args.out_painting:
        if random.random() < args.paint_rate:
            if random.random() < args.inpaint_rate:
                x = image_out_painting(x)
                y = image_out_painting(y)
    # if args.rand_noise:
    #     x = RandomNoising(x)
    #     y = RandomNoising(y)
    #
    # if args.rand_smooth:
    #     x = RandomSmoothing(x)
    #     y = RandomSmoothing(y)
    #
    # if args.rand_crop:
    #     x = RandomCropPad(x)
    #     y = RandomCropPad(y)

    # Save sample images module
    if args.save_samples is not None and status == "train" and random.random() < 0.01:
        n_sample = random.choice([i for i in range(args.batch_size)])
        sample_1 = np.concatenate(
            (x[n_sample, 0, :, :, 2 * img_deps // 6], y[n_sample, 0, :, :, 2 * img_deps // 6]), axis=1)
        sample_2 = np.concatenate(
            (x[n_sample, 0, :, :, 3 * img_deps // 6], y[n_sample, 0, :, :, 3 * img_deps // 6]), axis=1)
        sample_3 = np.concatenate(
            (x[n_sample, 0, :, :, 4 * img_deps // 6], y[n_sample, 0, :, :, 4 * img_deps // 6]), axis=1)
        sample_4 = np.concatenate(
            (x[n_sample, 0, :, :, 5 * img_deps // 6], y[n_sample, 0, :, :, 5 * img_deps // 6]), axis=1)
        final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
        final_sample = final_sample * 255.0
        final_sample = final_sample.astype(np.uint8)
        file_name = ''.join(
            [random.choice(string.ascii_letters + string.digits) for n in range(10)]) + '.' + args.save_samples
        imageio.imwrite(os.path.join(args.sample_path, args.exp_name, file_name), final_sample)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)
    # return x,y
