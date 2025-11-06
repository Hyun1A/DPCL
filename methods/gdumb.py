# When we make a new one, we should inherit the Finetune class.
import os, sys
import logging
import copy
import time
import datetime
import json
import random
import math
from collections import defaultdict




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

import ray
from configuration import config

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

args = config.base_parser()
if args.mode == 'gdumb':
    ray.init(num_gpus=args.num_gpus)


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i



class GDumb(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
        )
        self.memory_size = kwargs["memory_size"]
        self.n_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.batch_size = kwargs["batchsize"]
        self.n_tasks = kwargs["n_tasks"]
        self.eval_period = kwargs["eval_period"]
        self.eval_samples = []
        self.eval_time = []
        self.task_time = []
        
        self.iters = []
        self.eval_n_count_num = []
        self.is_end_task = []
        self.eval_exposed_classes = []
                  
        

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.exposed_classes.append(sample['klass'])
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(cls_list=self.exposed_classes)
        self.update_memory(sample)

    def update_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, end_task=False):
        
        print('task number:', self.cur_iter)
        print('number of seen data:', self.n_count_num)
        
        self.eval_samples.append(copy.deepcopy(self.memory.datalist))
        self.eval_time.append(sample_num)

        self.eval_n_count_num.append(self.n_count_num)
        self.is_end_task.append(end_task)
        self.iters.append(self.cur_iter)
        #print(self.memory.cls_list)
        #input()
        self.eval_exposed_classes.append(copy.deepcopy(self.memory.cls_list))

            
        dummy = {'avg_loss': 0.0, 'avg_acc': 0.0, 'cls_acc': np.zeros(self.n_classes)}   
        return dummy, dummy, dummy  
    
    

    def evaluate_all(self, test_list, n_epoch, batch_size, n_worker):
        
        eval_results = defaultdict(list)
        num_workers = args.num_gpus*args.workers_per_gpu
        num_evals = len(self.eval_samples)
        task_evals = [int(num_evals*i/self.n_tasks) for i in range(self.n_tasks)]
        task_records = defaultdict(list)
        
        print('num_evals:', num_evals)
        for i in range(math.ceil(num_evals/num_workers)):

            workers = [RemoteTrainer.remote(self.root, self.exp_name, self.model_name, self.dataset, self.n_classes, self.opt_name, self.lr,
                                            'cos', self.eval_samples[i*num_workers+j], test_list, self.criterion,
                                            self.train_transform, self.test_transform, self.cutmix,
                                            use_amp=self.use_amp, data_dir=self.data_dir, n_count_num=self.eval_n_count_num[i*num_workers+j],
                                            end_task=self.is_end_task[i*num_workers+j], cur_iter=self.iters[i*num_workers+j], mode=self.mode, seed=self.seed, tm=self.tm, memory_size=self.memory_size, 
                                            exposed_classes=self.eval_exposed_classes[i*num_workers+j],)
                                            for j in range(min(num_workers, num_evals-num_workers*i))]
            
            ray.get([workers[j].eval_worker.remote(n_epoch, batch_size, n_worker) for j in range(min(num_workers, num_evals-num_workers*i))])


    def after_task(self, cur_iter):
        pass


@ray.remote(num_gpus=1 / args.workers_per_gpu)
class RemoteTrainer:
    def __init__(self, root, exp_name, model_name, dataset, n_classes, opt_name, lr, sched_name, train_list, test_list,
                 criterion, train_transform, test_transform, cutmix, device=0, use_amp=False, data_dir=None, n_count_num=0, end_task=False, cur_iter=0, mode=None, seed=None, tm=None, memory_size=None, \
                 exposed_classes=None):

        self.root = root
        self.exp_name = exp_name

        self.n_count_num = n_count_num
        self.end_task = end_task
        self.cur_iter = cur_iter
        
        self.mode = mode
        self.seed = seed
        self.tm = tm
        self.memory_size = memory_size
        
        self.model_name = model_name
        self.dataset = dataset
        self.n_classes = n_classes

        self.exposed_classes = exposed_classes
        
        self.train_list = train_list
        self.test_list = test_list

        self.train_transform = train_transform
        self.test_transform = test_transform
        self.cutmix = cutmix

        self.exposed_classes = exposed_classes
        self.num_learned_class = len(self.exposed_classes)
        
        self.model = select_model(model_name, dataset, num_classes=self.num_learned_class)
        
        self.device = device
        self.model = self.model.cuda(self.device)
        self.criterion = criterion.cuda(self.device)
        self.topk = 1
        
        self.check_stream = 0
        

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.lr = lr
        # Initialize the optimizer and scheduler
        logger.info("Reset the optimizer and scheduler states")
        self.optimizer = select_optimizer(
            opt_name, self.lr, self.model, is_iBlurry=True
        )
        self.scheduler = select_scheduler(sched_name, self.optimizer)
        self.data_dir = data_dir

    def eval_worker(self, n_epoch, batch_size, n_worker):
        train_dataset = ImageDataset(
            self.root,
            pd.DataFrame(self.train_list),
            dataset=self.dataset,
            transform=self.train_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            preload=True,
            device=self.device,
            transform_on_gpu=True
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=min(len(self.train_list), batch_size),
            num_workers=n_worker,
        )

        self.model.train()

        for epoch in range(n_epoch):
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            total_loss, correct, num_data = 0.0, 0.0, 0.0

            idxlist = train_dataset.generate_idx(batch_size)
            for idx in idxlist:
                data = train_dataset.get_data_gpu(idx)
                x = data["image"]
                y = data["label"]

                x = x.cuda(self.device)
                y = y.cuda(self.device)

                self.optimizer.zero_grad()


                do_cutmix = self.cutmix and np.random.rand(1) < 0.5
                if do_cutmix:
                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit = self.model(x)
                            loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                    else:
                        logit = self.model(x)
                        loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                else:
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit = self.model(x)
                            loss = self.criterion(logit, y)
                    else:
                        logit = self.model(x)
                        loss = self.criterion(logit, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                loss = loss / x.size(0)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

            print(
                f"Task {self.cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {loss:.4f} | train_acc {correct/num_data :.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )            

            
        test_df = pd.DataFrame(self.test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            self.root,
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )

        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.cuda(self.device)
                y = y.cuda(self.device)
                logit = self.model(x)

                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = self.get_avg_res(total_num_data, total_loss, total_correct, correct_l, num_data_l)  
        
        return ret

    def get_avg_res(self, total_num_data, total_loss, total_correct, correct_l, num_data_l):
        total_loss = 0 if total_num_data == 0 else total_loss
        total_num_data = total_num_data if total_num_data > 0 else float('inf')
        
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        if self.check_stream == 1:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                   "exposed_classes": self.exposed_classes, "y":self.y.to('cpu').numpy()}
        
        else:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "exposed_classes": self.exposed_classes}
                    
        return ret

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        mask = (y==pred)
           
        correct_xlabel = y.masked_select(mask)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects
    
    
    def save_results(self, ret, end_task, islatest=False):
        if islatest:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', 'latest')
        else:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', self.tm)

        os.makedirs(folder_name, exist_ok=True)
        
        str_ = 'res_task_end_' if end_task else 'res_task_'        
        
        fn = os.path.join(folder_name, str_+str(self.cur_iter)+'_'+str(self.n_count_num)+'.pt')
        torch.save(ret, fn)

        if end_task:
            fn_ckpt = os.path.join(folder_name, 'model_task_'+str(self.cur_iter)+'.pt')
            torch.save(self.model.state_dict(), fn_ckpt)
            
