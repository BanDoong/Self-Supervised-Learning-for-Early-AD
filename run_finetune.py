import os
import numpy as np
import torch
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

# SimCLR

from simclr import SimCLR
from nt_xent import NT_Xent
# from simclr.modules.transformations import TransformsSimCLR
from sync_batchnorm import convert_model
from torchmetrics import ConfusionMatrix
import torch.nn as nn
import models_vit
from model import load_optimizer, save_model
import yaml
from operator import add
from timm.models.layers import trunc_normal_
# Dataset
from dataset import Dataset
from simclr import SimCLR_finetune
from utils import make_df, write_result
from torchvision import transforms
from preprocessing.clinica import MinMaxNormalization

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


def refine_keys(ckpt, num):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        if 'projector' in k:
            break
        name = k[num:]
        new_state_dict[name] = v
    return new_state_dict


def to_cpu(inputs):
    return inputs.cpu()


def cf_matrix(y_true, y_pred, num_label):
    cf = ConfusionMatrix(num_classes=num_label)
    _, y_pred = torch.max(y_pred, -1)
    y_pred = to_cpu(y_pred)
    y_true = to_cpu(y_true)
    return cf(y_true, y_pred).flatten()


def to_numpy(acc, spe, sen, f1, prec):
    acc = acc.numpy()
    spe = spe.numpy()
    sen = sen.numpy()
    f1 = f1.numpy()
    prec = prec.numpy()
    return acc, spe, sen, f1, prec


def add_element(cf, new):
    return torch.as_tensor(list(map(add, cf, new)))


def cal_metric(cf_list):
    TN, FN, FP, TP = cf_list
    print(cf_list)
    accuracy = (TP + TN) / (FP + FN + TP + TN) if FP + FN + TP + TN != 0 else torch.as_tensor(0.)
    # specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else torch.as_tensor(0.)
    precision = TP / (TP + FP) if (TP + FP) != 0 else torch.as_tensor(0.)
    # sensitivity or recall
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else torch.as_tensor(0.)
    bacc = (sensitivity + specificity) / 2
    # F1 = TP / (TP + (FN + FP) / 2) if TP + (FN + FP) / 2 != 0 else torch.as_tensor(0.)
    ppv = TP / (TP + FP) if (TP + FP) != 0 else torch.as_tensor(0.)
    npv = TN / (TN + FN) if (TN + FN) != 0 else torch.as_tensor(0.)

    if (precision + sensitivity) != 0:
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        F1 = torch.as_tensor(0.)
    accuracy, specificity, sensitivity, ppv, npv = to_numpy(accuracy, specificity, sensitivity, ppv, npv)
    bacc = bacc.numpy()
    return accuracy, bacc, specificity, sensitivity, ppv, npv


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)

    for step, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        # transformed =i & original =j
        # x_i = torch.as_tensor(batch_data[f'img_{args.modality}_i'], dtype=torch.float32)
        x_j = torch.as_tensor(batch_data[f'img_{args.modality}'], dtype=torch.float32)
        # x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)
        target = batch_data['label'].to(args.device)

        # positive pair, with encoding
        out = model(x_j)

        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
        cf_list = add_element(cf_list, new_cf_list)

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
        loss_epoch += [loss.item()]
    return np.mean(loss_epoch), cf_list


def val(args, val_loader, model, criterion, optimizer, writer):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)

    for step, batch_data in enumerate(val_loader):
        x_j = torch.as_tensor(batch_data[f'img_{args.modality}'], dtype=torch.float32)
        x_j = x_j.to(args.device, non_blocking=True)
        target = batch_data['label'].to(args.device)

        # positive pair, with encoding
        out = model(x_j)

        loss = criterion(out, target)

        new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
        cf_list = add_element(cf_list, new_cf_list)

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
        loss_epoch += [loss.item()]
    return np.mean(loss_epoch), cf_list


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transformation = transforms.Compose([MinMaxNormalization()])
    for fold in range(5):
        train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=transformation)
        val_dataset = Dataset(dataset='validation', fold=fold, args=args,transformation=transformation)
        if args.nodes > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size,
                                                                          rank=rank, shuffle=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=(val_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True
        )

        # initialize model
        ckpt = torch.load(os.path.join(args.model_path, f"checkpoint_{args.epoch_num}_{fold}.tar"),
                          map_location=args.device.type)
        if 'resnet' == args.model:
            from resnet import generate_model
            from simclr import SimCLR_res
            encoder = generate_model(model_depth=args.resnet_depth)
            model = SimCLR(encoder, args.projection_dim, n_features=encoder.fc.in_features)
            model.load_state_dict(ckpt)
            model = SimCLR_finetune(encoder, args)
            # trunc_normal_(model.mlp_head[1].weight, std=2e-5)

        elif 'clinica' == args.model:
            from model_clinica import Conv5_FC3_mni
            encoder = Conv5_FC3_mni()
            model = SimCLR(encoder, args.projection_dim, n_features=encoder.dim)
            model.load_state_dict(ckpt)
            model = SimCLR_finetune(encoder, args)
            # trunc_normal_(model.projector[0].weight, std=2e-5)
            # trunc_normal_(model.projector[2].weight, std=2e-5)
        else:
            encoder = models_vit.__dict__[args.model]()
            encoder = SimCLR(encoder, args.projection_dim, n_features=encoder.dim)
            model = SimCLR_finetune(encoder, args)
            model.load_state_dict(ckpt, strict=False)
        trunc_normal_(model.mlp_head[1].weight, std=2e-5)

        if args.model == 'conv_vit_base_patch16':
            ct = 0
            for child in model.children():
                ct += 1
                if ct > 4:
                    for param in child.parameters():
                        param.requires_grad = False

        if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Multi GPU ACTIVATES")
            model = nn.DataParallel(model)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=5, verbose=True,
                                                                   cooldown=10)
        criterion = torch.nn.CrossEntropyLoss()
        # DDP /DP
        if args.dataparallel:
            model = convert_model(model)
            model = DataParallel(model)
        else:
            if args.nodes > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[gpu])

        model = model.to(args.device)

        writer = None
        if args.nr == 0:
            writer = SummaryWriter()

        args.global_step = 0
        args.current_epoch = 0
        result_df = make_df(finetune=args.finetune)

        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr = optimizer.param_groups[0]['lr']
            loss_epoch, train_cf_list = train(args, train_loader, model, criterion, optimizer, writer)
            train_acc, train_bacc, train_spe, train_sen, train_f1, train_prec = cal_metric(train_cf_list)
            # if args.nr == 0 and scheduler:
            #     scheduler.step()

            # if args.nr == 0 and epoch % 30 == 0:
            #     save_model(args, model, optimizer)

            # if args.nr == 0:
            #     writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            #     writer.add_scalar("Misc/learning_rate", lr, epoch)
            #     print(
            #         f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            #     )
            #     args.current_epoch += 1

            with torch.no_grad():
                model.eval()
                val_loss_epoch, val_cf_list = val(args, val_loader, model, criterion, optimizer, writer)
                val_acc, val_bacc, val_spe, val_sen, val_f1, val_prec = cal_metric(val_cf_list)
                # accuracy, balnced accuracy, specificity, sensitivity, F1, precision
                print('\n')
                print('\n')
                print(f"Fold {fold}: epoch : {epoch}")
                print('\n')
                print(f'Loss : Train = {loss_epoch} | Val = {val_loss_epoch} ')
                print('\n')
                print("Train")
                print('\n')
                print(f'ACC : {train_acc}% | F1 : {train_f1}')
                print(f'BACC : {train_bacc}%')
                print(f'Specificity : {train_spe} | Sensitivity : {train_sen}')
                print(f'Precision : {train_prec}')
                print('\n')
                print("Validation")
                print('\n')
                print(f'ACC : {val_acc}% | F1 : {val_f1}')
                print(f'BACC : {val_bacc}%')
                print(f'Specificity : {val_spe} | Sensitivity : {val_sen}')
                print(f'Precision : {val_prec}')
            if args.scheduler:
                scheduler.step(train_acc)

            output_list = [epoch, loss_epoch, val_loss_epoch, train_acc, val_acc, train_bacc, val_bacc, train_spe,
                           val_spe, train_sen, val_sen, train_f1, val_f1, train_prec, val_prec]
            result_df = write_result(output_list, result_df)
        save_model(args, model, optimizer)
        result_df.to_csv(
            f'{args.model_path}/{args.modality}_mci_{args.mci}_epochs_{args.epoch_num}_results_fold_{fold}.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--config', default='./config/finetune.yaml', type=str)

    args = parser.parse_args()
    config = yaml_config_hook(args.config)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    for arg in vars(args):
        print(f'--{arg}', getattr(args, arg))

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
