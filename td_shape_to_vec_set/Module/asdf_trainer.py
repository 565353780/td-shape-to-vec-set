import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import optimization
from torch.optim import Adagrad as OPTIMIZER
from torch.optim.lr_scheduler import ReduceLROnPlateau as SCHEDULER

from td_shape_to_vec_set.Loss.edm import EDMLoss
from a_sdf.Method.path import createFileFolder, renameFile, removeFile
from a_sdf.Module.logger import Logger

from td_shape_to_vec_set.Dataset.asdf import ASDFDataset
from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond
from td_shape_to_vec_set.Method.time import getCurrentTime


def worker_init_fn(worker_id):    np.random.seed(
    np.random.get_state()[1][0] + worker_id)


class ASDFTrainer(object):
    def __init__(self):
        self.asdf_channel = 100
        self.sh_2d_degree = 4
        self.sh_3d_degree = 4
        self.channels = int(6 + (2*self.sh_2d_degree+1) + ((self.sh_3d_degree+1)**2))
        self.n_heads=8
        self.d_head=64
        self.depth=24

        self.batch_size = 64
        self.accumulation_steps = 1
        self.num_workers = 0
        self.lr = 1e-4
        self.weight_decay = 1e-10
        self.factor = 0.9
        self.patience = 1000
        self.min_lr = 1e-6
        self.warmup_epochs = 100
        self.train_epochs = 2000
        self.step = 0
        self.eval_step = 0
        self.loss_min = float('inf')
        self.eval_loss_min = float('inf')
        #'_warmup' + str(self.warmup_epochs) + \
        #'_train' + str(self.train_epochs) + \
        self.log_folder_name = getCurrentTime() + \
            '_lr' + str(self.lr) + \
            '_b' + str(self.batch_size * self.accumulation_steps) + \
            '_factor' + str(self.factor) + \
            '_patience' + str(self.patience) + \
            '_minlr' + str(self.min_lr) + \
            '_asdf' + str(self.asdf_channel) + \
            '_sh2d' + str(self.sh_2d_degree) + \
            '_sh3d' + str(self.sh_3d_degree) + \
            '_nheads' + str(self.n_heads) + \
            '_dheah' + str(self.d_head) + \
            '_depth' + str(self.depth)
        self.device = 'cuda'
        self.asdf_dataset_folder_path = '/home/chli/chLi/Dataset/ShapeNet/asdf/'

        self.model = EDMPrecond(
            n_latents=self.asdf_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth).to(self.device)
        '''
        self.model = ASDFAutoEncoder(
            asdf_channel=self.asdf_channel,
            sh_2d_degree=self.sh_2d_degree,
            sh_3d_degree=self.sh_3d_degree,
            hidden_dim=self.hidden_dim,
            dtype=torch.float32,
            device=self.device,
            sample_direction_num=self.sample_direction_num,
            direction_upscale=self.direction_upscale
        ).to(self.device)
        '''

        self.train_dataset = ASDFDataset(self.asdf_dataset_folder_path)
        # self.eval_dataset = PointsDataset(self.points_dataset_folder_path)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=self.num_workers,
                                           worker_init_fn=worker_init_fn)
        '''
        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=self.num_workers,
                                          worker_init_fn=worker_init_fn)
        '''

        '''
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        self.scheduler = optimization.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.warmup_epochs * len(self.train_dataloader) / self.accumulation_steps),
            num_training_steps=int(self.train_epochs * len(self.train_dataloader) / self.accumulation_steps),
            lr_end=1e-6,
            power=3)
        '''

        self.optimizer = OPTIMIZER(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = SCHEDULER(
            self.optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )

        self.logger = Logger()

        self.loss_func = EDMLoss()
        return

    def loadSummaryWriter(self):
        self.logger.setLogFolder("./logs/" + self.log_folder_name + "/")
        return True

    def loadModel(self, model_file_path, resume_model_only=False):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict= torch.load(model_file_path)

        self.model.load_state_dict(model_dict['model'])

        if not resume_model_only:
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.step= model_dict['step']
            self.eval_step= model_dict['eval_step']
            self.loss_min= model_dict['loss_min']
            self.eval_loss_min= model_dict['eval_loss_min']
            self.log_folder_name= model_dict['log_folder_name']

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict= {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'eval_step': self.eval_step,
            'loss_min': self.loss_min,
            'eval_loss_min': self.eval_loss_min,
            'log_folder_name': self.log_folder_name,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def getLr(self) -> float:
        return self.optimizer.state_dict()["param_groups"][0]["lr"]

    def trainStep(self, asdf_params, categories):
        self.model.train()

        loss = self.loss_func(self.model, asdf_params, categories)

        loss_item = loss.clone().detach().cpu().numpy()

        self.logger.addScalar("Train/loss", loss_item, self.step)

        if loss_item < self.loss_min:
            self.loss_min = loss_item
            self.saveModel("./output/" + self.log_folder_name +
                           "/model_best.pth")

        loss = loss / self.accumulation_steps
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1e2, norm_type=2)

        if self.step % self.accumulation_steps == 0:
            for params in self.model.parameters():
                params.grad[torch.isnan(params.grad)] = 0.0

            self.optimizer.step()

            self.model.zero_grad()
            self.optimizer.zero_grad()
        return loss_item

    def evalStep(self, data):
        self.model.eval()

        data = self.model(data)

        losses = data['losses']

        losses_tensor = torch.cat([
            loss if len(loss.shape) > 0 else loss.reshape(1)
            for loss in data['losses'].values()
        ])

        loss_sum = torch.sum(losses_tensor)
        loss_sum_float = loss_sum.detach().cpu().numpy()
        self.summary_writer.add_scalar("Eval/loss_sum", loss_sum_float,
                                       self.eval_step)

        if loss_sum_float < self.eval_loss_min:
            self.eval_loss_min = loss_sum_float
            self.saveModel("./output/" + self.log_folder_name +
                           "/model_eval_best.pth")

        for key, loss in losses.items():
            loss_tensor = loss.detach() if len(
                loss.shape) > 0 else loss.detach().reshape(1)
            loss_mean = torch.mean(loss_tensor)
            self.summary_writer.add_scalar("Eval/" + key, loss_mean,
                                           self.eval_step)
        return True

    def train(self, print_progress=False):
        total_epoch = 10000000

        self.model.zero_grad()
        for epoch in range(total_epoch):
            print("[INFO][Trainer::train]")
            print("\t start training, epoch : " + str(epoch + 1) + "/" +
                  str(total_epoch) + "...")
            if print_progress:
                pbar = tqdm(total=len(self.train_dataloader))
            for asdf_params, categories in self.train_dataloader:
                self.step += 1

                asdf_params = asdf_params.to(self.device, non_blocking=True)
                categories = categories.to(self.device, non_blocking=True)
                loss = self.trainStep(asdf_params, categories)

                if print_progress:
                    pbar.set_description(
                        "LOSS %.6f LR %.4f" % (loss, self.getLr())
                    )
                    pbar.update(1)

                self.logger.addScalar("Lr/lr", self.getLr(), self.step)

                if self.step % self.accumulation_steps == 0:
                    self.scheduler.step(loss)

            '''
            print("[INFO][Trainer::train]")
            print("\t start evaling, epoch : " + str(epoch + 1) + "/" +
                  str(total_epoch) + "...")
            for_data = self.eval_dataloader
            if print_progress:
                for_data = tqdm(for_data)
            #TODO: compute mean losses for one eval epoch
            for data in for_data:
                self.evalStep(data)
                self.eval_step += 1
            '''

            self.saveModel("./output/" + self.log_folder_name +
                           "/model_last.pth")
        return True
