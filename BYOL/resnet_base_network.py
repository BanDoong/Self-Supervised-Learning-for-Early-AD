import torch
import torch.nn as nn
from models.resnet import generate_model
import models.vit_model as vit_model
from model_clinica import Conv5_FC3


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class ResNet(torch.nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        resnet = generate_model(args.model_depth)
        # remove fc layer
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=args.mlp_hidden_size,
                                 projection_size=args.projection_size)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.projetion(h)


class ViT(nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        vit = vit_model.__dict__[args.model](classification=True)
        # remove fc layer
        self.encoder = torch.nn.Sequential(*list(vit.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projetion = MLPHead(in_channels=vit.mlp_head[1].in_features, mlp_hidden_size=args.mlp_hidden_size,
                                 projection_size=args.projection_size)
        self.batch_size = args.batch_size
        self.mlp_hidensize = args.mlp_hidden_size

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(self.batch_size, 768, -1)
        h = self.avgpool(h)
        # to batch*512 size
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
