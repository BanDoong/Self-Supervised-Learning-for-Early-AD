import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import torchio as tio
from dataset_multi import Dataset, MinMaxNormalization
from resnet_base_network import ResNet, MLPHead,ViT
from trainer import BYOLTrainer
from torchvision import transforms
import argparse
# from models.mlp_head import MLPHead
import random

print(torch.__version__)
from module import LARS


# torch.manual_seed(0)


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


class RandomFlip(object):
    def __call__(self, img):
        add_flip = tio.RandomFlip()
        return add_flip(img)


class RandomBlur(object):
    def __call__(self, img):
        add_blur = tio.RandomBlur()
        return add_blur(img)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def main(gpu, args):
    seed_everything(43)
    if 'caps' in args.dir_data:
        transformation = transforms.Compose([RandomFlip(), MinMaxNormalization()])
    else:
        transformation = transforms.Compose([RandomFlip(), RandomBlur()])

    fold = 0
    print(f'Fold is {fold}')

    train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=None)

    # online network
    if args.model == 'resnet':
        online_network = ResNet(args)
    elif 'vit' in args.model:
        online_network = ViT(args)
    pretrained_folder = args.model_path

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(args.device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        mlp_hidden_size=args.mlp_hidden_size, projection_size=args.projection_size)

    # target encoder
    if args.model == 'resnet':
        target_network = ResNet(args)
    elif 'vit' in args.model:
        target_network = ViT(args)

    if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Multi GPU ACTIVATES")
        online_network = nn.DataParallel(online_network)
        predictor = nn.DataParallel(predictor)
        target_network = nn.DataParallel(target_network)
    online_network = online_network.to(args.device)
    target_network = target_network.to(args.device)
    predictor = predictor.to(args.device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                    lr=args.lr * args.batch_size / 256, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, args.max_epochs, eta_min=0, last_epoch=-1        )
        # scheduler = None

    else:
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            list(online_network.parameters()) + list(predictor.parameters()),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

    # "decay the learning rate with the cosine decay schedule without restarts"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_epochs, eta_min=0, last_epoch=-1
    )

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          args=args,
                          scheduler=scheduler,
                          fold=fold)

    trainer.train(train_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BYOL")
    parser.add_argument('--config', default='./config/free64.yaml', type=str)

    args = parser.parse_args()
    config = yaml_config_hook(args.config)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    for arg in vars(args):
        print(f'--{arg}', getattr(args, arg))

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    main(0, args)
