import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import torchio as tio
from dataset_know import Dataset, MinMaxNormalization
from resnet_base_network import ResNet, ViT
from trainer_knowledge import BYOLTrainer
from torchvision import transforms
import argparse
from models.mlp_head import MLPHead
import random
from model_clinica import Conv5_FC3

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


# tranformation
class RandomFlip(object):
    def __call__(self, img):
        add_flip = tio.RandomFlip()
        return add_flip(img)


class RandomMotion(object):
    def __call__(self, img):
        add_motion = tio.RandomMotion()
        return add_motion(img)


class RandomSwap(object):
    def __call__(self, img):
        add_swap = tio.RandomSwap()
        return add_swap(img)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
        transformation = transforms.Compose([RandomFlip(), RandomSwap(), RandomMotion()])

    # tranformation flip,

    fold = 0
    print(f'Fold is {fold}')

    train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=None)

    # online network
    if args.model == 'resnet':
        online_network_1 = ResNet(args)
        online_network_2 = ResNet(args)
        shared_online = ResNet(args)
    elif 'vit' in args.model:
        online_network_1 = ViT(args)
        online_network_2 = ViT(args)
        shared_online = ViT(args)
    else:
        online_network_1 = Conv5_FC3().convolutions[:12]
        online_network_2 = Conv5_FC3().convolutions[:12]
        tmp = [Conv5_FC3().convolutions[12:], Conv5_FC3().flatten,
               MLPHead(in_channels=128 * 3 * 3 * 3, mlp_hidden_size=args.mlp_hidden_size,
                       projection_size=args.projection_size)]
        shared_online = nn.Sequential(*list(tmp))
    pretrained_folder = args.model_path
    # print(online_network_1)
    # print(shared_online)
    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(args.device)))

            online_network_1.load_state_dict(load_params['online_network_1_state_dict'])
            online_network_2.load_state_dict(load_params['online_network_2_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    if 'Conv' in args.model:
        predictor = MLPHead(in_channels=128,
                            mlp_hidden_size=args.mlp_hidden_size, projection_size=args.projection_size)
    else:
        predictor = MLPHead(in_channels=online_network_1.projetion.net[-1].out_features,
                            mlp_hidden_size=args.mlp_hidden_size, projection_size=args.projection_size)
    # target encoder
    if args.model == 'resnet':
        target_pre = ResNet(args)
        target_post = ResNet(args)
    elif 'vit' in args.model:
        target_pre = ViT(args)
        target_post = ViT(args)
    else:
        target_pre = Conv5_FC3().convolutions[:12]
        tmp = [Conv5_FC3().convolutions[12:], Conv5_FC3().flatten,
               MLPHead(in_channels=128 * 3 * 3 * 3, mlp_hidden_size=args.mlp_hidden_size,
                       projection_size=args.projection_size)]
        target_post = nn.Sequential(*list(tmp))

    if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Multi GPU ACTIVATES")
        online_network_1 = nn.DataParallel(online_network_1)
        online_network_2 = nn.DataParallel(online_network_2)
        shared_online = nn.DataParallel(shared_online)
        predictor = nn.DataParallel(predictor)
        target_pre = nn.DataParallel(target_pre)
        target_post = nn.DataParallel(target_post)

    online_network_1 = online_network_1.to(args.device)
    online_network_2 = online_network_2.to(args.device)
    shared_online = shared_online.to(args.device)
    target_pre = target_pre.to(args.device)
    target_post = target_post.to(args.device)
    predictor = predictor.to(args.device)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(list(online_network_1.parameters()) + list(predictor.parameters()) + list(
            online_network_2.parameters()) + list(shared_online.parameters()), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0,
                                                                           last_epoch=-1)
        # scheduler = None

    else:
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            list(online_network_1.parameters()) + list(predictor.parameters()) + list(
                online_network_2.parameters()) + list(shared_online.parameters()), lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epochs, eta_min=0, last_epoch=-1
        )

    trainer = BYOLTrainer(online_network_1=online_network_1,
                          online_network_2=online_network_2,
                          shared_online=shared_online,
                          target_pre=target_pre,
                          target_post=target_post,
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
