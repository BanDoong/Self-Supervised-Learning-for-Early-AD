import torch.nn as nn
import torch
from copy import deepcopy
from model_utils import *


class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """

    def __init__(self, dropout=0.5, n_classes=2, flatten_shape=128 * 3 * 3 * 3, after_=1300):
        super(Conv5_FC3, self).__init__()
        # fmt: off
        self.convolutions = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),

            # t1-linear : 128 * 6 * 7 * 6
            nn.Linear(flatten_shape, after_),
            nn.ReLU(),

            nn.Linear(after_, 50),
            nn.ReLU(),

            nn.Linear(50, n_classes)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]
        # fmt: on

    def forward(self, x):
        x = self.convolutions(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class Conv4_FC3(nn.Module):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, dropout=0.5, n_classes=2, flatten_shape=50 * 2 * 2 * 2):
        super(Conv4_FC3, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv3d(1, 15, 3, padding=0),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3, padding=0),
            nn.BatchNorm3d(25),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3, padding=0),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(50, 50, 3, padding=0),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(flatten_shape, 50),
            nn.ReLU(),

            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes)
        )

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


class AE_clinica(nn.Module):
    def __init__(self, model=None, mode=None, num_label=None, num_modality=None, pretrain=None, patch=None, roi=None):
        """
        Construct an autoencoder from a given CNN. The encoder part corresponds to the convolutional part of the CNN.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        """
        super(AE_clinica, self).__init__()
        self.mode = mode
        self.num_label = num_label
        self.num_modality = num_modality
        self.pretrain = pretrain
        self.patch = patch
        self.roi = roi
        self.level = 0

        if model is not None:
            self.encoder = deepcopy(model.convolutions)
            self.decoder = self.construct_inv_layers(model)

            for i, layer in enumerate(self.encoder):
                if isinstance(layer, PadMaxPool3d):
                    self.encoder[i].set_new_return()
                elif isinstance(layer, nn.MaxPool3d):
                    self.encoder[i].return_indices = True

        else:
            self.encoder = nn.Sequential()
            self.decoder = nn.Sequential()

    def encoder_out(self):

        conv = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        return conv

    def __len__(self):
        return len(self.encoder)

    def forward(self, x):

        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if isinstance(layer, PadMaxPool3d):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif isinstance(layer, nn.MaxPool3d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)

        return x

    def construct_inv_layers(self, model):
        """
        Implements the decoder part from the CNN. The decoder part is the symmetrical list of the encoder
        in which some layers are replaced by their transpose counterpart.
        ConvTranspose and ReLU layers are inverted in the end.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        :return: (Module) decoder part of the Autoencoder
        """
        inv_layers = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv3d):
                inv_layers.append(
                    nn.ConvTranspose3d(
                        layer.out_channels,
                        layer.in_channels,
                        layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
                )
                self.level += 1
            elif isinstance(layer, PadMaxPool3d):
                inv_layers.append(
                    CropMaxUnpool3d(layer.kernel_size, stride=layer.stride)
                )
            elif isinstance(layer, PadMaxPool2d):
                inv_layers.append(
                    CropMaxUnpool2d(layer.kernel_size, stride=layer.stride)
                )
            elif isinstance(layer, nn.Linear):
                inv_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, nn.Flatten):
                inv_layers.append(Reshape(model.flattened_shape))
            elif isinstance(layer, nn.LeakyReLU):
                inv_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
            else:
                inv_layers.append(deepcopy(layer))
        inv_layers = self.replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)

    @staticmethod
    def replace_relu(inv_layers):
        """
        Invert convolutional and ReLU layers (give empirical better results)

        :param inv_layers: (list) list of the layers of decoder part of the Auto-Encoder
        :return: (list) the layers with the inversion
        """
        idx_relu, idx_conv = -1, -1
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.ConvTranspose3d):
                idx_conv = idx
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                idx_relu = idx

            if idx_conv != -1 and idx_relu != -1:
                inv_layers[idx_relu], inv_layers[idx_conv] = (
                    inv_layers[idx_conv],
                    inv_layers[idx_relu],
                )
                idx_conv, idx_relu = -1, -1

        # Check if number of features of batch normalization layers is still correct
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.BatchNorm3d):
                conv = inv_layers[idx + 1]
                inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)

        return inv_layers
