import os
import torch
import numpy as np
from tqdm import trange
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from td_shape_to_vec_set.Loss.edm import EDMLoss
from td_shape_to_vec_set.Dataset.mash import MashDataset
from td_shape_to_vec_set.Dataset.single_shape import SingleShapeDataset
from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond

from ma_sh.Model.mash import Mash


class MashTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        device: str = "auto",
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_dataloader_x: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.context_dim = 512
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        self.channels = int(
            9 + (2 * self.mask_degree + 1) + ((self.sh_degree + 1) ** 2)
        )

        self.loss_func = EDMLoss()

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            device,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_dataloader_x,
        )
        return

    def createDatasets(self) -> bool:
        if False:
            mash_file_path = os.environ['HOME'] + '/Dataset/MashV4/ShapeNet/03636649/583a5a163e59e16da523f74182db8f2.npy'
            self.dataloader_dict['single_shape'] =  {
                'dataset': SingleShapeDataset(mash_file_path),
                'repeat_num': 1,
            }

        if True:
            self.dataloader_dict['mash'] =  {
                'dataset': MashDataset(self.dataset_root_folder_path, 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['image'] =  {
                'dataset': EmbeddingDataset(
                    self.dataset_root_folder_path,
                    {
                        'clip': 'Objaverse_82K/render_clip',
                        'dino': 'Objaverse_82K/render_dino',
                    },
                    'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['points'] =  {
                'dataset': EmbeddingDataset(self.dataset_root_folder_path, 'PointsEmbedding', 'train'),
                'repeat_num': 1,
            }

        if False:
            self.dataloader_dict['text'] =  {
                'dataset': EmbeddingDataset(self.dataset_root_folder_path, 'TextEmbedding_ShapeGlot', 'train'),
                'repeat_num': 10,
            }

        if True:
            self.dataloader_dict['eval'] =  {
                'dataset': MashDataset(self.dataset_root_folder_path, 'eval'),
            }

        return True

    def createModel(self) -> bool:
        self.model = EDMPrecond(
            n_latents=self.mash_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
            context_dim=self.context_dim,
        ).to(self.device)

        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if 'category_id' in data_dict.keys():
            data_dict['condition'] = data_dict['category_id']
        elif 'embedding' in data_dict.keys():
            data_dict['condition'] = data_dict['embedding']
        else:
            print('[ERROR][Trainer::toCondition]')
            print('\t valid condition type not found!')
            exit()

        noise, sigma, weight = self.loss_func(data_dict['mash_params'], not is_training)

        data_dict['noise'] = noise
        data_dict['sigma'] = sigma
        data_dict['weight'] = weight

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        inputs = data_dict['mash_params']
        D_yn = result_dict['D_x']
        weight = data_dict['weight']

        loss = weight * ((D_yn - inputs) ** 2)

        loss = loss.mean()

        loss_dict = {
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: torch.nn.Module, model_name: str) -> bool:
        sample_gt = False
        sample_num = 3
        timestamp_num = 18
        dataset = self.dataloader_dict['mash']['dataset']

        model.eval()

        data = dataset.__getitem__(0)
        gt_mash = data['mash_params']
        condition = data['category_id']

        if sample_gt:
            gt_mash = dataset.normalizeInverse(gt_mash)

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
        )[-1]

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

        if sample_gt and not self.gt_sample_added_to_logger:
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

            mash_params = dataset.normalizeInverse(mash_params)

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
