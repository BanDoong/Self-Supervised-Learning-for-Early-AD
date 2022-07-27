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
    def __init__(self, online_network_1, online_network_2, shared_online, target_pre, target_post, predictor, optimizer,
                 args,
                 fold, scheduler):
        self.online_network_1 = online_network_1
        self.online_network_2 = online_network_2
        self.shared_online = shared_online
        self.target_pre = target_pre
        self.target_post = target_post
        self.optimizer = optimizer
        self.device = args.device
        self.predictor = predictor
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
        mod_m = self.m / 2
        # for param_q, param_k in zip(self.online_network_1.parameters(), self.target_network.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q_1, param_q_2, param_k in zip(self.online_network_1.parameters(), self.online_network_2.parameters(),
                                                 self.target_pre.parameters()):
            param_k.data = param_k.data * self.m + param_q_1.data * (1. - mod_m) + param_q_2 * (1 - mod_m)
        for param_q, param_k in zip(self.shared_online.parameters(), self.target_post.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q_1, param_q_2, param_k in zip(self.online_network_1.parameters(), self.online_network_2.parameters(),
                                                 self.target_pre.parameters()):
            param_k.data = param_q_1.data / 2 + param_q_2.data / 2  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.shared_online.parameters(), self.target_post.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0

        self.initializes_target_network()
        before_loss = np.Inf

        if (self.args.device.type == 'cuda') and (torch.cuda.device_count() > 1):
            multigpu = True
        else:
            multigpu = False

        for epoch_counter in range(self.max_epochs + 1):
            current_loss = 0
            self.online_network_1.train()
            self.online_network_2.train()
            self.shared_online.train()
            for step, batch_data in enumerate(train_loader):
                batch_view_modality_1 = torch.as_tensor(batch_data[f'img_{self.modality[0]}'],
                                                        dtype=torch.float32).to(self.device)
                batch_view_modality_2 = torch.as_tensor(batch_data[f'img_{self.modality[1]}'],
                                                        dtype=torch.float32).to(self.device)

                loss = self.update(batch_view_modality_1, batch_view_modality_2)
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

    def update(self, batch_view_modality_1, batch_view_modality_2):

        # compute query feature
        predictions_from_modality_view_1 = self.predictor(
            self.shared_online(self.online_network_1(batch_view_modality_1)))
        predictions_from_modality_view_2 = self.predictor(
            self.shared_online(self.online_network_2(batch_view_modality_2)))

        # compute key features
        with torch.no_grad():
            targets_to_view_1 = self.target_post(self.target_pre(batch_view_modality_2))
            targets_to_view_2 = self.target_post(self.target_pre(batch_view_modality_1))

        loss = self.regression_loss(predictions_from_modality_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_modality_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH, multigpu=False):

        if multigpu:
            torch.save({
                'online_network_1_state_dict': self.online_network_1.module.state_dict(),
                'online_network_2_state_dict': self.online_network_2.module.state_dict(),
                'shared_online_state_dict': self.shared_online.module.state_dict(),
                'target_pre_state_dict': self.target_pre.module.state_dict(),
                'target_post_state_dict': self.target_pre.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
        else:
            torch.save({
                'online_network_1_state_dict': self.online_network_1.state_dict(),
                'online_network_2_state_dict': self.online_network_2.state_dict(),
                'shared_online_state_dict': self.shared_online.state_dict(),
                'target_pre_state_dict': self.target_pre.state_dict(),
                'target_post_state_dict': self.target_post.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)
