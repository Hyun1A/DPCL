# When we make a new one, we should inherit the Finetune class.
import os, sys
import logging
import copy
import time
import datetime
import json
from copy import deepcopy


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data
from utils.train_utils import select_model, select_optimizer, select_scheduler

from models.wgf import SVGD_kernal, SVGD_step, SGLD_step

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class DRO(ER):

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
        )

        self.method_dro = kwargs['method_dro']
        self.T_adv_dro = kwargs['T_adv_dro']
        self.gamma_dro = kwargs['gamma_dro']
        self.beta_dro = kwargs['beta_dro']
        self.stepsize_dro = kwargs['stepsize_dro']
    

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            
            if len(self.memory.images) > 0:
                train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                          iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            else:
                train_loss, train_acc = super().online_train(self.temp_batch, self.batch_size, n_worker,
                                                          iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            
            self.report_training(sample_num, train_loss, train_acc)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
        
    
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            
            x_stream, x_memory, y_stream, y_memory = None, None, None, None
            
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x_stream = stream_data['image']
                y_stream = stream_data['label']
                
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x_memory = memory_data['image']
                y_memory = memory_data['label']
                                
            x_stream = x_stream.to(self.device)
            y_stream = y_stream.to(self.device)
            
            x_memory = x_memory.to(self.device)
            y_memory = y_memory.to(self.device)
            
            self.optimizer.zero_grad()

            """"""
            rehearse = True
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x_stream, labels_a_stream, labels_b_stream, lam_stream = cutmix_data(x=x_stream, y=y_stream, alpha=1.0)
                x_memory, labels_a_memory, labels_b_memory, lam_memory = cutmix_data(x=x_memory, y=y_memory, alpha=1.0)
                
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        if rehearse:
                            z_hat = deepcopy(x_memory)
                            z_hat = z_hat.cuda()
                            z_hat = z_hat.clone().detach().requires_grad_(True)

                            for n in range(self.T_adv_dro):
                                delta = z_hat - x_memory
                                rho = torch.mean((torch.norm(delta.view(len(x_memory), -1), 2, 1) ** 2))
                                # loss_zt = F.cross_entropy(model(z_hat), y_memory)
                                loss_zt = lam_memory*self.criterion(self.model(z_hat), labels_a_memory) \
                                            + (1-lam_memory)*self.criterion(self.model(z_hat), labels_b_memory)
                                loss_zt /= labels_a_memory.size(0)
                                
                                loss_phi = - (loss_zt - self.gamma_dro * rho)
                                loss_phi.backward()
                                target_grad = z_hat.grad
                                deltanorm = torch.norm(delta, 2)

                                if self.method_dro == 'SVGD':
                                    input_shape = z_hat.size()
                                    flat_z = z_hat.view(input_shape[0], -1)
                                    target_grad = target_grad.view(input_shape[0], -1)
                                    flat_z = SVGD_step(self.stepsize_dro, flat_z, target_grad)
                                    z_hat = flat_z.view(list(input_shape))

                                elif self.method_dro == 'SGLD':
                                    z_hat = SGLD_step(self.stepsize, z_hat, target_grad)
                                z_hat = z_hat.clone().detach().requires_grad_(True)
                            self.optimizer.zero_grad()

                        loss_all = 0

                        updated_inds = None
                        logits - self.model(x_stream)
                        loss_a = lam_stream*self.criterion(logits, labels_a_stream) \
                                    + (1-lam_stream)*self.criterion(logits, labels_b_stream)
                        
                        loss_a = (loss_a).sum() / loss_a.size(0)

                        loss_all += loss_a

                        logits_adv = self.model(z_hat)
                        adv_loss = lam_memory*self.criterion(logits_adv, labels_a_memory) \
                                    + (1-lam_memory)*self.criterion(logits_adv, labels_b_memory)
                        adv_loss = adv_loss / labels_a_memory.size(0)
                        
                        logits_buffer = self.model(x_memory)
                        normal_loss = lam_memory*self.criterion(logits_buffer, labels_a_memory) \
                                    + (1-lam_memory)*self.criterion(logits_buffer, labels_b_memory)
                        normal_loss = normal_loss / labels_a_memory.size(0)
                        
                        total_loss = normal_loss + self.beta_dro*adv_loss

                        loss_all += total_loss

            else:
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # logit = self.model(x)
                        # loss = self.criterion(logit, y)
                        if rehearse:
                            z_hat = deepcopy(x_memory)
                            z_hat = z_hat.cuda()
                            z_hat = z_hat.clone().detach().requires_grad_(True)

                            for n in range(self.T_adv_dro):
                                delta = z_hat - x_memory
                                rho = torch.mean((torch.norm(delta.view(len(x_memory), -1), 2, 1) ** 2))
                                loss_zt = self.criterion(self.model(z_hat), y_memory) / y_memory.size(0)
                                loss_phi = - (loss_zt - self.gamma_dro * rho)
                                loss_phi.backward()
                                target_grad = z_hat.grad
                                deltanorm = torch.norm(delta, 2)


                                if self.method_dro == 'SVGD':
                                    input_shape = z_hat.size()
                                    flat_z = z_hat.view(input_shape[0], -1)
                                    target_grad = target_grad.view(input_shape[0], -1)
                                    flat_z = SVGD_step(self.stepsize_dro, flat_z, target_grad)
                                    z_hat = flat_z.view(list(input_shape))

                                elif self.method_dro == 'SGLD':
                                    z_hat = SGLD_step(self.stepsize, z_hat, target_grad)
                                z_hat = z_hat.clone().detach().requires_grad_(True)
                            self.optimizer.zero_grad()

                        loss_all = 0

                        logits = self.model(x_stream)
                        loss_a = self.criterion(logits, y_stream) / y_stream.size(0)

                        loss_all += loss_a

                        logits_adv = self.model(z_hat)
                        adv_loss = self.criterion(logits_adv, y_memory) / y_memory.size(0)

                        logits_buffer = self.model(x_memory)
                        normal_loss = self.criterion(logits_buffer, y_memory) / y_memory.size(0)
                        total_loss = normal_loss + self.beta_dro*adv_loss

                        loss_all += total_loss
                
                
            #loss = loss/x.size(0)

            loss = loss_all
            y = torch.cat([y_stream, y_memory], dim=0)
            logit = torch.cat([logits, logits_buffer], dim=0)
            
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

            """"""
            
        return total_loss / iterations, correct / num_data
