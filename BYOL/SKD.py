import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import write_result, make_df


class BYOLTrainer:
    def __init__(self, online_network_1, online_network_2, target_network_1, target_network_2, predictor_1, predictor_2,
                 optimizer, args,
                 fold, scheduler):
        self.online_network_1 = online_network_1
        self.online_network_2 = online_network_2
        self.target_network_1 = target_network_1
        self.target_network_2 = target_network_2
        self.optimizer = optimizer
        self.device = args.device
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        self.max_epochs = args.max_epochs
        self.writer = SummaryWriter()
        # exponential moving avg parameter for momentum update
        self.m = args.m
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.checkpoint_interval = args.checkpoint_interval
        self.modality = args.modality
        self.model_path = args.model_path
        self.scheduler = scheduler
        self.fold = fold
        self.args = args
        self.result_df = make_df(False)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network_1.parameters(), self.target_network_1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.online_network_2.parameters(), self.target_network_2.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network_1.parameters(), self.target_network_1.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.online_network_2.parameters(), self.target_network_2.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0

        self.initializes_target_network()
        before_loss = np.Inf

        if (self.args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            multigpu = True

        for epoch_counter in range(self.max_epochs + 1):
            current_loss = 0
            for step, batch_data in enumerate(train_loader):
                batch_view_modality_1_1 = torch.as_tensor(batch_data[f'img_{self.modality[0]}_1'],
                                                          dtype=torch.float32).to(
                    self.device)
                batch_view_modality_1_2 = torch.as_tensor(batch_data[f'img_{self.modality[0]}_2'],
                                                          dtype=torch.float32).to(
                    self.device)
                batch_view_modality_2_1 = torch.as_tensor(batch_data[f'img_{self.modality[1]}_1'],
                                                          dtype=torch.float32).to(
                    self.device)
                batch_view_modality_2_2 = torch.as_tensor(batch_data[f'img_{self.modality[1]}_2'],
                                                          dtype=torch.float32).to(
                    self.device)

                loss = self.update(batch_view_modality_1_1, batch_view_modality_1_2, batch_view_modality_2_1,
                                   batch_view_modality_2_2)
                print(f"Fold : {self.fold}\t Step [{step}/{len(train_loader)}]\t Loss: {loss}")
                current_loss += loss
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
            if self.scheduler:
                self.scheduler.step()
                niter += 1
            # last_loss = current_loss.cpu().numpy()
            print("End of epoch {}".format(epoch_counter))
            # output_list = [epoch_counter, last_loss]
            # self.result_df = write_result(output_list, self.result_df)
            # save checkpoints
            if epoch_counter % self.args.checkpoint_interval == 0:
                self.save_model(os.path.join(f'{self.model_path}/model_{epoch_counter}_fold_{self.fold}.pth'), multigpu)
            if before_loss >= current_loss:
                before_loss = current_loss
                self.save_model(os.path.join(f'{self.model_path}/model_best_weights_fold_{self.fold}.pth'), multigpu)
                # os.system(f'{print(epoch_counter)} >> best_epochs.txt')

    def update(self, batch_view_modality_1_1, batch_view_modality_1_2, batch_view_modality_2_1,
                                   batch_view_modality_2_2):

        # compute query feature
        predictions_from_modality_view_1_1 = self.predictor_1(self.online_network_1(batch_view_modality_1_1))
        predictions_from_modality_view_1_2 = self.predictor_1(self.online_network_1(batch_view_modality_1_2))
        predictions_from_modality_view_2_1 = self.predictor_1(self.online_network_1(batch_view_modality_1_1))
        predictions_from_modality_view_2_2 = self.predictor_1(self.online_network_1(batch_view_modality_1_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_1_1 = self.target_network_1(batch_view_modality_1_2)
            targets_to_view_1_2 = self.target_network_1(batch_view_modality_1_1)
            targets_to_view_2_1 = self.target_network_2(batch_view_modality_2_2)
            targets_to_view_2_2 = self.target_network_2(batch_view_modality_2_1)

        loss = self.regression_loss(predictions_from_modality_view_1_1, targets_to_view_1_1)
        loss += self.regression_loss(predictions_from_modality_view_1_2, targets_to_view_1_2)
        loss += self.regression_loss(predictions_from_modality_view_2_1, targets_to_view_2_1)
        loss += self.regression_loss(predictions_from_modality_view_2_2, targets_to_view_2_2)
        return loss.mean()

    def save_model(self, PATH, multigpu=False):

        if multigpu:
            torch.save({
                'online_network_1_state_dict': self.online_network_1.module.state_dict(),
                'online_network_2_state_dict': self.online_network_2.module.state_dict(),
                'target_network_1_state_dict': self.target_network_1.module.state_dict(),
                'target_network_2_state_dict': self.target_network_2.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
        else:
            torch.save({
                'online_network_1_state_dict': self.online_network_1.state_dict(),
                'online_network_2_state_dict': self.online_network_2.state_dict(),
                'target_network_1_state_dict': self.target_network_1.state_dict(),
                'target_network_2_state_dict': self.target_network_2.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
