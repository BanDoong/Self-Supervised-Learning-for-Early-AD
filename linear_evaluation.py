import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from dataset import Dataset
from simclr import SimCLR
from logistic_regression import LogisticRegression
import models_vit
from run import yaml_config_hook


def inference(args, loader, simclr_model):
    feature_vector = []
    labels_vector = []
    for step, batch_data in enumerate(loader):
        x = torch.as_tensor(batch_data[f'img_{args.modality}'], dtype=torch.float32)
        x = x.to(args.device, non_blocking=True)
        y = batch_data['label'].to(args.device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        # feature_vector.extend(h.numpy())
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.cpu().detach().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, args):
    train_X, train_y = inference(args, train_loader, simclr_model)
    test_X, test_y = inference(args, test_loader, simclr_model)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            model.zero_grad()

            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)

            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc

            loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


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

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(5):
        train_dataset = Dataset(dataset='train', fold=fold, args=args)
        val_dataset = Dataset(dataset='validation', fold=fold, args=args)
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
            batch_size=args.logistic_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
        )

        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.logistic_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.workers,
        )

        if 'resnet' == args.model:
            from resnet import generate_model

            encoder = generate_model(model_depth=50)
            n_features = encoder.fc.in_features
        else:
            encoder = models_vit.__dict__[args.model]()
            n_features = encoder.dim  # get dimensions of fc layer
        simclr_model = SimCLR(encoder, args.projection_dim, n_features)
        # load pre-trained model from checkpoint
        ckpt = torch.load(os.path.join(args.model_path, f"checkpoint_{args.epoch_num}.tar"),
                          map_location=args.device.type)
        simclr_model.load_state_dict(ckpt)
        if (args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print("Multi GPU ACTIVATES")
            simclr_model = nn.DataParallel(simclr_model)
        simclr_model = simclr_model.to(args.device)
        simclr_model.eval()

        ## Logistic Regression
        n_classes = 2
        model = LogisticRegression(n_features, n_classes)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        print("### Creating features from pre-trained context model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            simclr_model, train_loader, test_loader, args
        )

        arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, args.logistic_batch_size
        )

        for epoch in range(args.logistic_epochs):
            loss_epoch, accuracy_epoch = train(
                args, arr_train_loader, simclr_model, model, criterion, optimizer
            )
            print(
                f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

        # final testing
        loss_epoch, accuracy_epoch = test(
            args, arr_test_loader, simclr_model, model, criterion, optimizer
        )
        print(
            f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
        )
