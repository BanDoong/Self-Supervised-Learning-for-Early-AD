import torch
import torch.nn as nn


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand < frac] = 0
    return data_noise


def CBR_nopad_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    layers = []
    layers += [
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=bias)]
    layers += [nn.BatchNorm3d(num_features=out_channels)]
    layers += [nn.LeakyReLU()]
    cbr = nn.Sequential(*layers)
    return cbr


def CBR3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    layers = []
    layers += [
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=bias)]
    layers += [nn.BatchNorm3d(num_features=out_channels)]
    layers += [nn.LeakyReLU()]
    layers += [PadMaxPool3d(2, 2, return_indices=True, return_pad=True)]
    cbr = nn.Sequential(*layers)
    return cbr


def CB3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    layers = []
    layers += [
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=bias)]
    layers += [nn.BatchNorm3d(num_features=out_channels)]
    cb = nn.Sequential(*layers)
    return cb


def DCBR3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    layers = []
    layers += [CropMaxUnpool3d(2, 2)]
    layers += [
        nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias)]
    layers += [nn.BatchNorm3d(num_features=out_channels)]
    layers += [nn.LeakyReLU()]
    cbr = nn.Sequential(*layers)
    return cbr


def buildNetwork(layers, encoder, activation="relu"):
    net = []
    if encoder:
        for i in range(1, len(layers)):
            net.append(CBR3d(layers[i - 1], layers[i]))
    else:
        for i in range(1, len(layers)):
            net.append(DCBR3d(layers[i - 1], layers[i]))
    return nn.Sequential(*net)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(*self.size)


class PadMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad2d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(2)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[1], 0, coords[0], 0)
            else:
                return output


class CropMaxUnpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool2d, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[2]
            y1 = padding[0]
            output = output[:, :, x1::, y1::]

        return output


class PadMaxPool3d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad3d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(3)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[2], 0, coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[2], 0, coords[1], 0, coords[0], 0)
            else:
                return output


class CropMaxUnpool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool3d, self).__init__()
        self.unpool = nn.MaxUnpool3d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[4]
            y1 = padding[2]
            z1 = padding[0]
            output = output[:, :, x1::, y1::, z1::]

        return output
