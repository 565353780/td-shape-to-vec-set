import os
import torch
import numpy as np
from math import sqrt, ceil
from tqdm import tqdm
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Module.o3d_viewer import O3DViewer
from ma_sh.Module.local_editor import LocalEditor

from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond


class MashSampler(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = True,
        device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.context_dim = 512
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24

        self.use_ema = use_ema
        self.device = device

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

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def toInitialMashModel(self) -> Mash:
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
        return mash_model

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][MashSampler::loadModel]")
            print("\t model_file not exist!")
            print('\t model_file_path:', model_file_path)
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][MashSampler::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        diffuse_steps: int = 18,
    ) -> torch.Tensor:
        self.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][Sampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        sampled_array = self.model.sample(
            cond=condition_tensor,
            batch_seeds=torch.arange(0, sample_num).to(self.device),
            diffuse_steps=diffuse_steps,
        ).float()

        return sampled_array

    @torch.no_grad()
    def step_sample(
        self,
        sample_num: int,
        condition: Union[int, np.ndarray] = 0,
        diffuse_steps: int = 18,
    ) -> bool:
        self.model.eval()

        if isinstance(condition, int):
            condition_tensor = torch.ones([sample_num]).long().to(self.device) * condition
        elif isinstance(condition, np.ndarray):
            # condition dim: 1x768
            condition_tensor = torch.from_numpy(condition).type(torch.float32).to(self.device).repeat(sample_num, 1)
        else:
            print('[ERROR][Sampler::sample]')
            print('\t condition type not valid!')
            return np.ndarray()

        object_dist = [2, 0, 2]

        row_num = ceil(sqrt(sample_num))

        print("start diffuse", sample_num, "mashs....")
        sampled_array = self.model.sample(
            cond=condition_tensor,
            batch_seeds=torch.arange(0, sample_num).to(self.device),
            diffuse_steps=diffuse_steps,
            step_sample=True,
        )

        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()
        o3d_viewer.update()

        mash_model = self.toInitialMashModel()
        for i in range(diffuse_steps + 1):
            print("start create mash points for diffuse step Itr." + str(i) + "...")

            o3d_viewer.clearGeometries()

            mash_pcd_list = []
            for j in tqdm(range(sample_num)):
                mash_params = sampled_array[i][j]

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

                translate = [
                    int(j / row_num) * object_dist[0],
                    0 * object_dist[1],
                    (j % row_num) * object_dist[2],
                ]

                pcd.translate(translate)
                mash_pcd_list.append(pcd)

            o3d_viewer.addGeometries(mash_pcd_list)
            o3d_viewer.update()

        o3d_viewer.run()
        return True
