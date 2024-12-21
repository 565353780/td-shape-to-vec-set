import os
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm, trange
from copy import deepcopy
from typing import Union
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from td_shape_to_vec_set.Loss.edm import EDMLoss
from td_shape_to_vec_set.Dataset.mash import MashDataset
from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond
from td_shape_to_vec_set.Method.path import createFileFolder, renameFile, removeFile
from td_shape_to_vec_set.Method.time import getCurrentTime
from td_shape_to_vec_set.Module.logger import Logger

from ma_sh.Model.mash import Mash


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def check_and_replace_nan_in_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradient: {name}")
            param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
    return True

class MashTrainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "cuda:0",
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.local_rank = setup_distributed()

        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.context_dim = 512
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        self.accum_iter = accum_iter
        if device == 'auto':
            self.device = torch.device('cuda:' + str(self.local_rank))
        else:
            self.device = device
        self.warm_step_num = warm_step_num / accum_iter
        self.finetune_step_num = finetune_step_num
        self.lr = lr * batch_size / 256 * self.accum_iter * dist.get_world_size()
        self.ema_start_step = ema_start_step
        self.ema_decay_init = ema_decay_init
        self.ema_decay = ema_decay

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path

        self.step = 0

        self.logger = None
        if self.local_rank == 0:
            self.logger = Logger()

        self.dataloader_dict = {}

        if True:
            self.dataloader_dict['mash'] =  {
                'dataset': MashDataset(dataset_root_folder_path, 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['image'] =  {
                'dataset': EmbeddingDataset(
                    dataset_root_folder_path,
                    {
                        'clip': 'Objaverse_82K/render_clip',
                        'dino': 'Objaverse_82K/render_dino',
                    },
                    'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['points'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'PointsEmbedding', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['text'] =  {
                'dataset': EmbeddingDataset(dataset_root_folder_path, 'TextEmbedding_ShapeGlot', 'train'),
                'repeat_num': 10,
            }

        for key, item in self.dataloader_dict.items():
            self.dataloader_dict[key]['sampler'] = DistributedSampler(item['dataset'])
            self.dataloader_dict[key]['dataloader'] = DataLoader(
                item['dataset'],
                sampler=self.dataloader_dict[key]['sampler'],
                batch_size=batch_size,
                num_workers=num_workers,
            )

        self.channels = int(
            9 + (2 * self.mask_degree + 1) + ((self.sh_degree + 1) ** 2)
        )
        self.model = EDMPrecond(
            n_latents=self.mash_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
            context_dim=self.context_dim,
        ).to(self.device)

        if self.local_rank == 0:
            self.ema_model = deepcopy(self.model)
            self.ema_loss = None

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.sched = LambdaLR(self.optim, lr_lambda=self.warmup_lr)

        self.loss_func = EDMLoss()

        self.initRecords()

        self.gt_sample_added_to_logger = False
        return

    def initRecords(self) -> bool:
        if self.logger is None:
            return True

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path)
        if 'model' in model_state_dict.keys():
            self.model.module.load_state_dict(model_state_dict["model"])
        if 'ema_model' in model_state_dict.keys():
            self.ema_model.load_state_dict(model_state_dict["ema_model"])
        if 'ema_loss' in model_state_dict.keys():
            self.ema_loss = model_state_dict['ema_loss']
        if 'step' in model_state_dict.keys():
            self.step = model_state_dict['step']

        return True

    def getLr(self) -> float:
        return self.optim.state_dict()["param_groups"][0]["lr"]

    def warmup_lr(self, step: int) -> float:
        return min(step, self.warm_step_num) / self.warm_step_num

    def toEMADecay(self) -> float:
        if self.step <= self.ema_start_step:
            return self.ema_decay_init + self.step / self.ema_start_step * (self.ema_decay - self.ema_decay_init)

        return self.ema_decay

    def ema(self) -> bool:
        ema_decay = self.toEMADecay()

        source_dict = self.model.module.state_dict()
        target_dict = self.ema_model.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * ema_decay + source_dict[key].data * (1 - ema_decay)
            )
        return True

    def trainStep(
        self,
        data: dict,
    ) -> dict:
        self.model.train()

        mash_params = data['mash_params'].to(self.device)
        condition = data['condition']
        if isinstance(condition, torch.Tensor):
            condition = condition.to(self.device)
        else:
            for key in condition.keys():
                condition[key] = condition[key].to(self.device)

        self.model.train()

        loss = self.loss_func(self.model, mash_params, condition)

        loss_item = loss.clone().detach().cpu().numpy()

        accum_loss = loss / self.accum_iter

        accum_loss.backward()

        if not check_and_replace_nan_in_grad(self.model):
            print('[ERROR][Trainer::trainStep]')
            print('\t check_and_replace_nan_in_grad failed!')
            exit()

        if (self.step + 1) % self.accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            self.sched.step()
            if self.local_rank == 0:
                self.ema()
            self.optim.zero_grad()

        loss_dict = {
            "Loss": loss_item,
        }

        return loss_dict


    @torch.no_grad()
    def sampleModelStep(self, model: torch.nn.Module, model_name: str) -> bool:
        if self.local_rank != 0:
            return True

        model.eval()

        sample_num = 3
        timestamp_num = 36
        data = self.dataloader_dict['mash']['dataset'].__getitem__(0)
        gt_mash = data['mash_params']
        condition = data['category_id']

        print('[INFO][Trainer::sampleModelStep]')
        print("\t start diffuse", sample_num, "mashs....")
        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        elif isinstance(condition, dict):
            condition_tensor = {}
            for key in condition.keys():
                condition_tensor[key] = condition[key].type(torch.float32).to(self.device).unsqueeze(0).repeat(sample_num, *([1] * condition[key].dim()))
        else:
            print('[ERROR][Trainer::sampleModelStep]')
            print('\t condition type not valid!')
            return False

        sampled_array = model.sample(
            cond=condition_tensor,
            batch_seeds=torch.arange(0, sample_num).to(self.device),
            diffuse_steps=timestamp_num,
        ).float()

        mash_model = Mash(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            20,
            800,
            0.4,
            dtype=torch.float64,
            device=self.device,
        )

        if not self.gt_sample_added_to_logger:
            sh2d = 2 * self.mask_degree + 1
            ortho_poses = gt_mash[:, :6]
            positions = gt_mash[:, 6:9]
            mask_params = gt_mash[:, 9 : 9 + sh2d]
            sh_params = gt_mash[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud('GT_MASH/gt_mash', pcd, self.step)

            self.gt_sample_added_to_logger = True

        for i in trange(sample_num):
            mash_params = sampled_array[i]

            sh2d = 2 * self.mask_degree + 1
            ortho_poses = mash_params[:, :6]
            positions = mash_params[:, 6:9]
            mask_params = mash_params[:, 9 : 9 + sh2d]
            sh_params = mash_params[:, 9 + sh2d :]

            mash_model.loadParams(
                mask_params=mask_params,
                sh_params=sh_params,
                positions=positions,
                ortho6d_poses=ortho_poses
            )

            pcd = mash_model.toSamplePcd()

            self.logger.addPointCloud(model_name + '/pcd_' + str(i), pcd, self.step)

        return True

    @torch.no_grad()
    def sampleStep(self) -> bool:
        self.sampleModelStep(self.model.module, 'Model')
        return True

    @torch.no_grad()
    def sampleEMAStep(self) -> bool:
        self.sampleModelStep(self.ema_model, 'EMA')
        return True

    def toCondition(self, data: dict) -> Union[torch.Tensor, None]:
        if 'category_id' in data.keys():
            return data['category_id']

        if 'embedding' in data.keys():
            return data["embedding"]

        print('[ERROR][Trainer::toCondition]')
        print('\t valid condition type not found!')
        return None

    def train(self) -> bool:
        final_step = self.step + self.finetune_step_num

        if self.local_rank == 0:
            print("[INFO][Trainer::train]")
            print("\t start training ...")

        loss_dict_list = []

        epoch_idx = 1
        while self.step < final_step or self.finetune_step_num < 0:
            self.model.train()

            for data_name, dataloader_dict in self.dataloader_dict.items():
                dataloader_dict['sampler'].set_epoch(epoch_idx)

                dataloader = dataloader_dict['dataloader']
                repeat_num = dataloader_dict['repeat_num']

                for i in range(repeat_num):
                    if self.local_rank == 0:
                        print('[INFO][Trainer::train]')
                        print('\t start training on dataset [', data_name, '] ,', i + 1, '/', repeat_num, '...')

                    if self.local_rank == 0:
                        pbar = tqdm(total=len(dataloader))
                    for data in dataloader:
                        condition = self.toCondition(data)
                        if condition is None:
                            print('[ERROR][Trainer::train]')
                            print('\t toCondition failed!')
                            continue

                        conditional_data = {
                            'mash_params': data['mash_params'],
                            'condition': condition,
                        }

                        train_loss_dict = self.trainStep(conditional_data)

                        loss_dict_list.append(train_loss_dict)

                        lr = self.getLr()

                        if (self.step + 1) % self.accum_iter == 0 and self.local_rank == 0:
                            for key in train_loss_dict.keys():
                                value = 0
                                for i in range(len(loss_dict_list)):
                                    value += loss_dict_list[i][key]
                                value /= len(loss_dict_list)
                                self.logger.addScalar("Train/" + key, value, self.step)
                            self.logger.addScalar("Train/Lr", lr, self.step)

                            if self.ema_loss is None:
                                self.ema_loss = train_loss_dict["Loss"]
                            else:
                                ema_decay = self.toEMADecay()

                                self.ema_loss = self.ema_loss * ema_decay + train_loss_dict["Loss"] * (1 - ema_decay)
                            self.logger.addScalar("Train/EMALoss", self.ema_loss, self.step)

                            loss_dict_list = []

                        if self.local_rank == 0:
                            pbar.set_description(
                                "EPOCH %d LOSS %.6f LR %.4f"
                                % (
                                    epoch_idx,
                                    train_loss_dict["Loss"],
                                    self.getLr() / self.lr,
                                )
                            )

                        self.step += 1
                        if self.local_rank == 0:
                            pbar.update(1)

                    if self.local_rank == 0:
                        pbar.close()

                if self.local_rank == 0:
                    self.autoSaveModel("total")

                if self.local_rank == 0:
                    if epoch_idx % 1 == 0:
                        self.sampleStep()
                        # self.sampleEMAStep()

                epoch_idx += 1

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.module.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "ema_loss": self.ema_loss,
            "step": self.step,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, name: str) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + name + "_model_last.pth"

        tmp_save_last_model_file_path = save_last_model_file_path[:-4] + "_tmp.pth"

        self.saveModel(tmp_save_last_model_file_path)

        removeFile(save_last_model_file_path)
        renameFile(tmp_save_last_model_file_path, save_last_model_file_path)

        return True
