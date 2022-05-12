import torch
import torch.nn as nn
from models_vit import vit_base_patch16, vit_huge_patch14, vit_large_patch16


class VitSimCLR(nn.Module):

    def __init__(self, args):
        super(VitSimCLR, self).__init__()

        self.backbone = args.model
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.LeakyReLU(), self.backbone.fc)

    def _get_basemodel(self, model):
        return model

    def forward(self, x):
        return self.backbone(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.mlp_head = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.encoder.classifier = nn.Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


class SimCLR_finetune(nn.Module):
    def __init__(self, encoder, args):
        super(SimCLR_finetune, self).__init__()

        self.encoder = encoder
        if args.model =='resnet':
            if args.resnet_depth == 50:
                self.encoder.dim = 2048
            else:
                self.encoder.dim = 512
        else:
            self.encoder.dim = 768

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.encoder.dim),
            nn.Linear(self.encoder.dim, args.num_label)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(self.encoder)
        return self.mlp_head(x)


class SimCLR_notransfer(nn.Module):
    def __init__(self, encoder, args):
        super(SimCLR_notransfer, self).__init__()

        self.encoder = encoder
        self.encoder.fc = nn.Identity()
        if args.resnet_depth == 50:
            self.encoder.dim = 2048
        else:
            self.encoder.dim = 512

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.encoder.dim),
            nn.Linear(self.encoder.dim, args.num_label)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.mlp_head(x)