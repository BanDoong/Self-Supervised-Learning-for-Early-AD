import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
# distributed training
from torch.nn.parallel import DataParallel
# Tensorboard
from torch.utils.tensorboard import SummaryWriter
from model_clinica import AE_clinica, Conv5_FC3
from torchvision import transforms
import yaml

# Dataset
from dataset import Dataset, MinMaxNormalization
from run_finetune import cf_matrix, add_element, cal_metric
from utils import make_df, write_result


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', multigpu=False, pretrain=False,
                 f1=None):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.f1_min = np.Inf
        self.delta = delta
        self.path = path
        self.pretrain = pretrain
        self.multigpu = multigpu
        self.f1 = f1

    def __call__(self, val_loss, model, f1):

        if self.pretrain:
            score = f1
        else:
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.pretrain:
                self.save_checkpoint(f1=f1, model=model)
            else:
                self.save_checkpoint(val_loss=val_loss, model=model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.pretrain:
                self.save_checkpoint(f1=f1, model=model)
            else:
                self.save_checkpoint(val_loss=val_loss, model=model)
            self.counter = 0

    def save_checkpoint(self, val_loss=None, f1=None, model=None):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            if self.pretrain:
                print(f'BACC Score is Increased ({self.f1_min} --> {f1}).  Saving model ...')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.multigpu:
            torch.save(model.module.state_dict(), self.path)
            print("==========================================")
        else:
            torch.save(model.state_dict(), self.path)
        if self.pretrain:
            self.f1_min = f1
        else:
            self.val_loss_min = val_loss


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


def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)
    if args.finetune:
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            x = batch_data[f'img_{args.modality}'].to(args.device)
            target = batch_data['label'].to(args.device)
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
            cf_list = add_element(cf_list, new_cf_list)
            loss_epoch += [loss.item()]
        return np.mean(loss_epoch), cf_list
    else:
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            x = batch_data[f'img_{args.modality}'].to(args.device)
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            loss_epoch += [loss.item()]
        return np.mean(loss_epoch)


def val(args, val_loader, model, criterion):
    loss_epoch = []
    cf_list = [0] * (args.num_label ** 2)
    model.eval()
    with torch.no_grad():
        if args.finetune:
            for step, batch_data in enumerate(val_loader):
                x = batch_data[f'img_{args.modality}'].to(args.device)
                target = batch_data['label'].to(args.device)
                out = model(x)
                loss = criterion(out, target)
                new_cf_list = cf_matrix(y_true=target, y_pred=out, num_label=args.num_label)
                cf_list = add_element(cf_list, new_cf_list)
                loss_epoch += [loss.item()]
            return np.mean(loss_epoch), cf_list
        else:
            for step, batch_data in enumerate(val_loader):
                x = batch_data[f'img_{args.modality}'].to(args.device)
                out = model(x)
                loss = criterion(out, x)
                loss_epoch += [loss.item()]
            return np.mean(loss_epoch)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class extract_conv(nn.Module):
    def __init__(self, num_label=2, model=None, dropout=0.5):
        super(extract_conv, self).__init__()
        self.encoder = model.encoder_out()
        self.fc = nn.Sequential(Flatten(),
                                nn.Dropout(p=dropout),
                                nn.Linear(128 * 6 * 7 * 6, 1300),
                                nn.ReLU(),
                                nn.Linear(1300, 50),
                                nn.ReLU(),
                                nn.Linear(50, num_label))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


def ae_generate_sampler(dataset):
    from torch.utils.data import sampler
    df = dataset.subj_list
    weights = [1] * len(df) * 1

    sampler.RandomSampler(weights)


def clas_generate_sampler(dataset):
    from torch.utils.data import sampler
    df = dataset.subj_list
    n_labels = 2
    count = np.zeros(n_labels)

    for idx in df.index:
        label = df['diagnosis'][idx]
        if label == 'CN':
            count[0] += 1
        elif label == 'MCI':
            count[1] += 1

    weight_per_class = 1 / np.array(count)
    weights = []

    for idx, label in enumerate(dataset['diagnosis']):
        if label == 'CN':
            weights += [weight_per_class[0]]
        elif label == 'MCI':
            weights += [weight_per_class[1]]

    return sampler.RandomSampler(weights)

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result_df = make_df(args.finetune)

    min_max_transform = transforms.Compose([MinMaxNormalization()])
    for fold in range(5):
        print(f'Fold is {fold}')
        train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=min_max_transform)
        val_dataset = Dataset(dataset='validation', fold=fold, args=args, transformation=min_max_transform)
        if args.finetune:
            train_sampler = clas_generate_sampler(train_dataset)
        else:
            train_sampler = ae_generate_sampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=pl_worker_init_function,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            pin_memory=True
        )

        # initializer model
        model = AE_clinica(model=Conv5_FC3())
        if args.finetune:
            ckpt = torch.load(f'{args.model_path}/model_image_{fold}.pt')
            model.load_state_dict(ckpt)
            print('Pretrained is loaded')
            model = extract_conv(model=model)

        model = model.to(args.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.finetune:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        if args.finetune:
            save_path = f'{args.model_path}/finetune_model_image_{fold}.pt'
        else:
            save_path = f'{args.model_path}/model_image_{fold}.pt'
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path, multigpu=False,
                                       pretrain=args.finetune)
        for epoch in range(args.start_epoch, args.epochs):
            print(f"Epoch : {epoch}")
            model.train()

            if args.finetune:
                loss_epoch, train_cf_list = train(args, train_loader, model, criterion, optimizer)
                train_acc, train_bacc, train_spe, train_sen, train_ppv, train_npv = cal_metric(train_cf_list)
                val_loss_epoch, val_cf_list = val(args, val_loader, model, criterion)
                val_acc, val_bacc, val_spe, val_sen, val_ppv, val_npv = cal_metric(val_cf_list)
                # accuracy, balnced accuracy, specificity, sensitivity, F1, precision
                print('\n')
                print('\n')
                print(f"Fold {fold}: epoch : {epoch}")
                print('\n')
                print(f'Loss : Train = {loss_epoch} | Val = {val_loss_epoch} ')
                print('\n')
                print("Train")
                print('\n')
                print(f'ACC : {train_acc} ')
                print(f'BACC : {train_bacc}')
                print(f'Specificity : {train_spe} | Sensitivity : {train_sen}')
                print(f'PPV : {train_ppv}')
                print(f'NPV : {train_npv}')
                print('\n')
                print("Validation")
                print('\n')
                print(f'ACC : {val_acc}')
                print(f'BACC : {val_bacc}')
                print(f'Specificity : {val_spe} | Sensitivity : {val_sen}')
                print(f'PPV : {val_ppv}')
                print(f'NPV : {val_npv}')
                early_stopping(val_loss=None, model=model, f1=val_bacc)
                output_list = [epoch, loss_epoch, val_loss_epoch, train_acc, val_acc, train_bacc, val_bacc, train_spe,
                               val_spe, train_sen, val_sen, train_ppv, val_ppv, train_npv, val_npv]
            else:
                loss_epoch = train(args, train_loader, model, criterion, optimizer)
                val_loss_epoch = val(args, val_loader, model, criterion)
                print(f'Training Loss : {loss_epoch}')
                print(f'Validation Loss : {val_loss_epoch}')
                early_stopping(val_loss=val_loss_epoch, model=model, f1=None)
                output_list = [epoch, loss_epoch, val_loss_epoch]

            result_df = write_result(output_list, result_df)

            if epoch == args.epochs or early_stopping.early_stop:
                break
        if args.finetune:
            result_df.to_csv(f'{args.model_path}/finetune_{args.modality}_cnmci_results_fold_{fold}.csv')
        else:
            result_df.to_csv(f'{args.model_path}/{args.modality}_cnmci_results_fold_{fold}.csv')


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

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(0, args)
