# When we make a new one, we should inherit the Finetune class.
import os, sys
import logging
import copy
import time
import datetime
import json


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

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ER_Multi_SWAG(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, writer, **kwargs
        )
        
        self.root = kwargs["root"]
        self.exp_name = kwargs["exp_name"]
        self.tm = kwargs["tm"]
        self.seed = kwargs["rnd_seed"]
        self.mode = kwargs["mode"]
        self.cur_iter = 0
        self.n_count_num = 0
        self.writer = writer
        
        self.num_learned_class = 0
        
        self.num_learning_class = 1
        self.n_classes = n_classes
        
        self.exposed_classes = []
        self.check_stream = 0
        
        self.seen = 0
        self.topk = kwargs["topk"]
        
        self.device = device
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'exp_reset'
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.memory_size = kwargs["memory_size"]
        self.memory_size_total = kwargs["memory_size"]
        
        self.data_dir = kwargs["data_dir"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        #self.memory_size -= self.temp_batchsize

        self.gpu_transform = kwargs["gpu_transform"]
        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.9999
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        self.memory = MemoryDataset(self.root, self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform)
                
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167, 'imagenet_subset': 128741*2, 'imagenet_subset_sub_shuffle': 128741*2, 'cifar100_hier_setup1':100000,  'cifar100_hier_setup2':50000, 'stanford_car_setup1':8144*2, 'imagenet_subset_setup2': 128741, 'stanford_car_setup2':8144}
        self.total_samples = num_samples[self.dataset]
        
        
        """ For multi-swag """
        self.n_ens_fcs = 5
        
        model_cfg_kwargs={'a':None}
        model_cfg_args=[self.model.fc.in_features, self.num_learned_class]
        
        for i in range(self.n_ens_fcs):
            swags_fc = SWAG(nn.Linear, 
                    subspace_type=kwargs['subspace'], subspace_kwargs={'max_rank': args.max_num_models},
                    *model_cfg_args, num_classes=1, **model_cfg_kwargs)
            
            
        
        self.swag_fcs = []
        
        
        

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            self.report_training(sample_num, train_loss, train_acc)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

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
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x,y)

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

            
            """ Do SWAG """
            n_ensembled = 0
            if self.swag and (self.n_count_num + 1) > self.swag_start and (self.n_count_num + 1 - self.swag_start) % self.swag_c_iter == 0:
                for i, fc in enumerate(self.model.fcs):
                    n_ensembled += 1
                    self.swag_fcs[i].collect_model(fc) # for deviation
            

        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y):
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
                
        loss = loss/x.size(0)
        

        return logit, loss

    def report_training(self, sample_num, train_loss, train_acc):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, end_task=False):
        test_df = pd.DataFrame(test_list)
        
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        
        test_dataset = ImageDataset(
            self.root,
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
         
        eval_dict = self.evaluation(test_loader, self.criterion, end_task)
        
          
        print('logs')
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        
        return eval_dict

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model, is_iBlurry=True)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def evaluation(self, test_loader, criterion, end_task):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        
        self.swag_fcs[i].collect_model(fc)
        
        
        
        self.swag_model.set_swa()
        self.swag_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.swag_model(x)

                loss = criterion(logit, y)
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

        print('save result for task'+str(self.cur_iter+1))
        self.save_results(ret, end_task)
        self.save_results(ret, end_task, islatest=True)
            
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
            
