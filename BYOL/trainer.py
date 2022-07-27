import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import write_result, make_df
import math


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, args, fold, scheduler):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = args.device
        self.predictor = predictor
        self.max_epochs = args.max_epochs
        self.writer = SummaryWriter()
        # exponential moving avg parameter for momentum update
        self.base_m = args.m
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.checkpoint_interval = args.checkpoint_interval
        self.total_step = 0
        self.modality = args.modality
        self.model_path = args.model_path
        self.scheduler = scheduler
        self.fold = fold
        self.args = args
        self.result_df = make_df(False)
        # self.target_ema_updater = EMA(self.m)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # update_moving_average(self.target_ema_updater, self.target_network, self.online_network)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    # def set_requires_grad(self, model, val):
    #     for p in model.parameters():
    #         p.requires_grad = val

    def _decay_ema_momentum(self, step):
        m = (1 - (1 - self.base_m) * (math.cos(math.pi * step / self.total_step) + 1) / 2)
        return m

    # def paper_loss(self, x, y):
    #     norm_x, norm_y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    #     return -2. * torch.mean(torch.sum(x * y, dim=-1) / (norm_x * norm_y))
    @torch.no_grad()
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        # model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        from main import cosine_scheduler
        # ============ init schedulers ... ============
        # #base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
        # base_value = self.args.lr * self.args.batch_size / 256.
        # final_value = 1e-6
        # lr_schedule = cosine_scheduler(base_value=base_value, final_value=final_value, epochs=self.args.max_epochs,
        #                                niter_per_ep=len(train_loader), warmup_epochs=10)
        # wd_schedule = cosine_scheduler(0.04, 0.4, self.args.max_epochs, len(train_loader))
        # # # momentum parameter is increased to 1. during training with a cosine schedule
        # momentum_schedule = cosine_scheduler(self.args.m, 1, self.args.max_epochs, len(train_loader))

        self.initializes_target_network()
        self.total_step = self.max_epochs * train_dataset.num_data // self.batch_size
        before_loss = np.Inf
        # os.system('touch best_epochs.txt')
        # self.total_global_step = len(train_loader) * epoch_counter + step - 1

        if (self.args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            multigpu = True
        else:
            multigpu = False
        step_T = 0
        self.m = self.base_m
        for epoch_counter in range(self.max_epochs + 1):
            current_loss = 0
            for step, batch_data in enumerate(train_loader):
                batch_view_1 = batch_data[f'img_{self.modality}_i'].to(self.device)
                batch_view_2 = batch_data[f'img_{self.modality}_j'].to(self.device)

                loss = self.update(batch_view_1, batch_view_2)
                print(f"Fold : {self.fold}\t Step [{step}/{len(train_loader)}]\t Loss: {loss}")
                current_loss += loss
                self.writer.add_scalar('loss', loss, global_step=step_T)

                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                # update the key encoder
                # global_step = len(train_loader) * epoch_counter + step_T - 1
                if self.scheduler:
                    self.scheduler.step()
                self._update_target_network_parameters()
                self.m = self._decay_ema_momentum(step_T)
                step_T += 1
            # last_loss = current_loss.cpu().numpy()
            print("End of epoch {}".format(epoch_counter))
            # output_list = [epoch_counter, last_loss]
            # self.result_df = write_result(output_list, self.result_df)
            # save checkpoints
            if epoch_counter % self.args.checkpoint_interval == 0:
                self.save_model(
                    os.path.join(f'{self.model_path}/model_{self.args.modality}_{epoch_counter}_fold_{self.fold}.pth'),
                    multigpu)
            if before_loss >= current_loss:
                before_loss = current_loss
                self.save_model(
                    os.path.join(f'{self.model_path}/model_{self.args.modality}_best_weights_fold_{self.fold}.pth'),
                    multigpu)
                # os.system(f'{print(epoch_counter)} >> best_epochs.txt')

    def update(self, batch_view_1, batch_view_2):

        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH, multigpu=False):
        if multigpu:
            torch.save({
                'online_network_state_dict': self.online_network.module.state_dict(),
                'target_network_state_dict': self.target_network.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
        else:
            torch.save({
                'online_network_state_dict': self.online_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
