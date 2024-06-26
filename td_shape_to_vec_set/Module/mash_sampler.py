import os
import torch
from math import sqrt, ceil
from tqdm import tqdm
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud
from ma_sh.Module.o3d_viewer import O3DViewer

from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond


class MashSampler(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.sh_2d_degree = 3
        self.sh_3d_degree = 2
        self.channels = int(
            6 + (2 * self.sh_2d_degree + 1) + ((self.sh_3d_degree + 1) ** 2)
        )
        self.n_heads = 1
        self.d_head = 64
        self.depth = 12

        self.device = device

        self.model = EDMPrecond(
            n_latents=self.mash_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
        ).to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def toInitialMashModel(self) -> Mash:
        mash_model = Mash(
            self.mash_channel,
            self.sh_2d_degree,
            self.sh_3d_degree,
            dtype=torch.float32,
            device=self.device,
        )
        return mash_model

    def loadModel(self, model_file_path, resume_model_only=True):
        if not os.path.exists(model_file_path):
            print("[ERROR][MashSampler::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        self.model.load_state_dict(model_dict["model"])

        if not resume_model_only:
            # self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        print("[INFO][MashSampler::loadModel]")
        print("\t load model success!")
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        diffuse_steps: int,
        category_id: int = 0,
    ) -> torch.Tensor:
        self.model.eval()

        sampled_array = self.model.sample(
            cond=torch.Tensor([category_id] * sample_num).long().to(self.device),
            batch_seeds=torch.arange(0, sample_num).to(self.device),
            diffuse_steps=diffuse_steps,
        ).float()

        return sampled_array

    @torch.no_grad()
    def step_sample(
        self,
        sample_num: int,
        diffuse_steps: int,
        category_id: int = 0,
    ) -> bool:
        self.model.eval()

        object_dist = [2, 0, 2]

        row_num = ceil(sqrt(sample_num))

        print("start diffuse", sample_num, "mashs....")
        sampled_array = self.model.sample(
            cond=torch.Tensor([category_id] * sample_num).long().to(self.device),
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

                sh2d = 2 * self.sh_2d_degree + 1

                rotation_vectors = mash_params[:, :3]
                positions = mash_params[:, 3:6]
                mask_params = mash_params[:, 6 : 6 + sh2d]
                sh_params = mash_params[:, 6 + sh2d :]

                mash_model.loadParams(
                    mask_params, sh_params, rotation_vectors, positions
                )
                mash_points = toNumpy(torch.vstack(mash_model.toSamplePoints()[:2]))
                pcd = getPointCloud(mash_points)

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
