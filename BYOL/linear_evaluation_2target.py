import torch
import torch.nn as nn
from dataset import Dataset
from resnet_base_network import ResNet, ViT
import argparse
import os
from main import yaml_config_hook
from torch.utils.data import DataLoader
import numpy as np
from main import RandomFlip, seed_everything
from torchvision import transforms
from utils import make_df, write_result, cal_metric, cf_matrix, add_element
from model_clinica import Conv5_FC3


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # self.linear = torch.nn.Linear(input_dim, output_dim)
        self.linear = torch.nn.Linear(input_dim, 128)
        self.batch = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU()
        self.linear_2 = nn.Linear(128, output_dim)
        self.apply_init()

    def apply_init(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.xavier_normal_(self.linear_2.weight)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch(x)
        x = self.relu(x)
        return self.linear_2(x)


def get_features_from_encoder(encoder, loader, device):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for step, batch_data in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(batch_data['img_mri'].to(device))
            x_train.extend(feature_vector)
            y_train.extend(batch_data['label'].to(device))

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size * 8, shuffle=False)
    return train_loader, test_loader


def refine_keys(ckpt):
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in ckpt.items():
        k = k[7:]
        new_state[k] = v
    return new_state


def main(gpu, args):
    seed_everything(43)
    transformation = transforms.Compose([RandomFlip()])
    for fold in range(5):
        result_df = make_df(finetune=True)
        print(f'Fold is {fold}')

        train_dataset = Dataset(dataset='train', fold=fold, args=args, transformation=None)
        val_dataset = Dataset(dataset='val', fold=fold, args=args, transformation=None)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, drop_last=False, shuffle=True)
        test_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False, shuffle=True)
        if args.model == 'resnet':
            encoder = ResNet(args)
            output_feature_dim = encoder.projetion.net[0].in_features
        elif 'vit' in args.model:
            encoder = ViT(args)
            output_feature_dim = 512
        else:
            encoder = Conv5_FC3().convolutions
            output_feature_dim = 128

        if args.load_best:
            weight_path = f'{args.model_path}/model_best_weights_fold_0.pth'
        else:
            weight_path = f'{args.model_path}/model_{args.loading_epoch_num}_fold_0.pth'
            # load pre-trained parameters
        load_params = torch.load(weight_path, map_location=torch.device(torch.device(args.device)))

        if 'online_network_state_dict' in load_params:
            load_params['online_network_1_state_dict'] = load_params['online_network_state_dict']
            encoder.load_state_dict(load_params['online_network_1_state_dict'])
            print("Parameters successfully loaded.")

        # remove the projection head
        if args.model == 'resnet' or 'vit' in args.model:
            encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
        # default output_feature_dim = 512
        logreg = LogisticRegression(output_feature_dim, 2)
        if torch.cuda.device_count() > 1:
            encoder = nn.DataParallel(encoder)
            logreg = nn.DataParallel(logreg)
        encoder = encoder.to(args.device)
        logreg = logreg.to(args.device)

        encoder.eval()
        x_train, y_train = get_features_from_encoder(encoder, train_loader, args.device)
        x_test, y_test = get_features_from_encoder(encoder, test_loader, args.device)

        if len(x_train.shape) > 2:
            if 'vit' in args.model:
                x_train = torch.mean(x_train, dim=2)
                x_test = torch.mean(x_test, dim=2)
            else:
                x_train = torch.mean(x_train, dim=[2, 3, 4])
                x_test = torch.mean(x_test, dim=[2, 3, 4])

        print("Training data shape:", x_train.shape, y_train.shape)
        print("Testing data shape:", x_test.shape, y_test.shape)

        # scaler = preprocessing.StandardScaler()
        # scaler.fit(x_train)
        # x_train = scaler.transform(x_train).astype(np.float32)
        # x_test = scaler.transform(x_test).astype(np.float32)
        train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train,
                                                                    x_test, y_test,
                                                                    batch_size=args.batch_size)
        # encoder.train()
        # optimizer = torch.optim.Adam(list(logreg.parameters())+list(encoder.parameters()), lr=3e-4)
        # for linear evaluation
        optimizer = torch.optim.SGD(logreg.parameters(), args.lr * args.batch_size / 256., momentum=0.9,
                                    weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0,
                                                               last_epoch=-1)
        # adam and T_max = args.max_epoch  70.4 acc
        # adam and T_max = 100 68 acc
        criterion = torch.nn.CrossEntropyLoss()
        eval_every_n_epochs = args.eval_every_n_epochs

        for epoch in range(args.max_epochs):
            cf_list = [0] * (args.num_label ** 2)
            for x, y in train_loader:
                x = x.to(args.device)
                y = y.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                logits = logreg(x)
                predictions = torch.argmax(logits, dim=1)

                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()
                scheduler.step()

            total = 0
            if epoch % eval_every_n_epochs == 0:
                correct = 0
                for x, y in test_loader:
                    x = x.to(args.device)
                    y = y.to(args.device)

                    logits = logreg(x)
                    predictions = torch.argmax(logits, dim=1)

                    new_cf_list = cf_matrix(y_true=y, y_pred=logits, num_label=args.num_label)
                    cf_list = add_element(cf_list, new_cf_list)

                    total += y.size(0)
                    correct += (predictions == y).sum().item()
                acc, bacc, spe, sen, ppv, npv, f1 = cal_metric(cf_list)
                # acc = 100 * correct / total
                # print(f"Testing accuracy: {np.mean(acc)}")
                print(f'ACC : {acc}')
                print(f'BACC : {bacc}')
                print(f'Specificity : {spe} | Sensitivity : {sen}')
                print(f'PPV : {ppv}')
                print(f'NPV : {npv}')
                print(f'F1 : {f1}')
                output_list = [epoch, acc, bacc, spe, sen, ppv, npv, f1]

                result_df = write_result(output_list, result_df)
        if args.load_best:
            result_df.to_csv(f'{args.model_path}/finetune_{args.modality}_cnmci_results_fold_best_{fold}.csv')
        else:
            result_df.to_csv(
                f'{args.model_path}/finetune_{args.modality}_cnmci_results_fold_{args.loading_epoch_num}_{fold}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BYOL")
    parser.add_argument('--config', default='./config/out_free64.yaml', type=str)

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
