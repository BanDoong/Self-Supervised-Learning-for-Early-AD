import os
import numpy as np
import torch
import torch.nn as nn
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from preprocessing.clinica import RandomNoising, RandomSmoothing, RandomCropPad, MinMaxNormalization
# SimCLR

from simclr import SimCLR
from nt_xent import NT_Xent
# from simclr.modules.transformations import TransformsSimCLR
from sync_batchnorm import convert_model

import models_vit
from model import load_optimizer, save_model
import yaml

# Dataset
from dataset import Dataset


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


def refine_keys(ckpt):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = []
    for step, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        if args.trans_tau:
            x_i = torch.as_tensor(batch_data[f'img_{args.modality[0]}_i'], dtype=torch.float32)
            x_j = torch.as_tensor(batch_data[f'img_{args.modality[1]}_j'], dtype=torch.float32)
        else:
            x_i = torch.as_tensor(batch_data[f'img_{args.modality}_i'], dtype=torch.float32)
            x_j = torch.as_tensor(batch_data[f'img_{args.modality}_j'], dtype=torch.float32)
        x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)
        # for nt_xent
        loss = criterion(z_i, z_j)
        # loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))
        if args.nr == 0 and step % 50 == 0:
            print(f'Steop [{step}/{len(train_loader)}] \t Loss : {loss.item()}')
        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += [loss.item()]
    return np.mean(loss_epoch)


def pl_worker_init_function(worker_id: int) -> None:  # pragma: no cover
    """
    The worker_init_fn that Lightning automatically adds to your dataloader if you previously set
    set the seed with ``seed_everything(seed, workers=True)``.
    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.
    """

    def _get_rank() -> int:
        """Returns 0 unless the environment specifies a rank."""
        rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
        for key in rank_keys:
            rank = os.environ.get(key)
            if rank is not None:
                return int(rank)
        return 0

    global_rank = _get_rank()
    # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64 if torch.__version__ > "1.7.0" else np.uint32
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (
            stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]
    ).sum()
    random.seed(stdlib_seed)


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    transformation_list = [MinMaxNormalization()]
    if args.rand_noise:
        transformation_list.append(RandomNoising())
    if args.rand_smooth:
        transformation_list.append(RandomSmoothing())
    if args.rand_crop:
        transformation_list.append(RandomCropPad())

    transformation = transforms.Compose(transformation_list)
    for fold in range(5):
        print(f'Fold is {fold}')

        train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=transformation)
        if args.nodes > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
            )
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True
        )

        # initializer model

        if 'resnet' == args.model:
            from resnet import generate_model
            encoder = generate_model(model_depth=args.resnet_depth)
            n_features = encoder.fc.in_features
            # ckpt = torch.load('./resnet_50_23dataset.pth')
            # encoder.load_state_dict(refine_keys(ckpt['state_dict']), strict=False)
        elif 'clinica' == args.model:
            from model_clinica import Conv5_FC3_mni
            encoder = Conv5_FC3_mni()
            n_features = 128 * 4 * 5 * 4
        else:
            encoder = models_vit.__dict__[args.model](classification=False)
            n_features = encoder.dim

        # initialize model
        model = SimCLR(encoder, args.projection_dim, n_features)
        if args.reload:
            model_fp = os.path.join(
                args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
            )
            model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

        if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Multi GPU ACTIVATES")
            model = nn.DataParallel(model)

        model = model.to(args.device)

        optimizer, scheduler = load_optimizer(args, model)
        criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

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

        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr = optimizer.param_groups[0]['lr']
            loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

            if args.nr == 0 and scheduler:
                scheduler.step()

            if args.nr == 0 and epoch % 100 == 0:
                save_model(args, model, fold)

            if args.nr == 0:
                writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
                writer.add_scalar("Misc/learning_rate", lr, epoch)
                print(
                    f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
                )
                args.current_epoch += 1

        save_model(args, model, fold)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument('--config', default='./config/config.yaml', type=str)

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
