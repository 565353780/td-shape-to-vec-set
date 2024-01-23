import os
import torch
from math import sqrt, ceil
from tqdm import tqdm
from typing import Union

from data_convert.Method.data import toData

from a_sdf.Model.asdf_model import ASDFModel
from a_sdf.Method.pcd import getPointCloud
from a_sdf.Method.render import renderGeometries
from a_sdf.Module.o3d_viewer import O3DViewer

from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond


class ASDFSampler(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.asdf_channel = 100
        self.sh_2d_degree = 4
        self.sh_3d_degree = 4
        self.n_heads=8
        self.d_head=64
        self.depth=24
        self.sample_direction_num = 200
        self.direction_upscale = 4

        self.device = device

        self.model = EDMPrecond(
            n_latents=self.asdf_channel,
            channels=int(6 + (2*self.sh_2d_degree+1) + ((self.sh_3d_degree+1)**2)),
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth).to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def toInitialASDFModel(self) -> ASDFModel:
        asdf_model = ASDFModel(
            max_sh_3d_degree=self.sh_3d_degree,
            max_sh_2d_degree=self.sh_2d_degree,
            use_inv=False,
            dtype=torch.float32,
            device="cpu",
            sample_direction_num=self.sample_direction_num,
            direction_upscale=self.direction_upscale,
        )

        return asdf_model

    def loadModel(self, model_file_path, resume_model_only=True):
        if not os.path.exists(model_file_path):
            print("[ERROR][ASDFSampler::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path)

        self.model.load_state_dict(model_dict["model"])

        if not resume_model_only:
            # self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        print("[INFO][ASDFSampler::loadModel]")
        print("\t load model success!")
        return True

    @torch.no_grad()
    def sample(
        self, sample_num: int, diffuse_steps: int, rad_density: int, category_id: int=0
    ) -> bool:
        self.model.eval()

        object_dist = [2, 0, 2]

        row_num = ceil(sqrt(sample_num))

        asdf_pcd_list = []

        print("start diffuse", sample_num, 'asdfs....')
        sampled_array = self.model.sample(
            cond=torch.Tensor([category_id] * sample_num).long().to(self.device),
            batch_seeds=torch.arange(0, sample_num).to(
                self.device
            ),
            diffuse_steps=diffuse_steps,
        ).float()

        print(
            sampled_array.shape,
            sampled_array.max(),
            sampled_array.min(),
            sampled_array.mean(),
            sampled_array.std(),
        )

        asdf_model = self.toInitialASDFModel()

        for i in tqdm(range(sample_num)):
            asdf_model.loadParams(sampled_array[i])
            asdf_points = asdf_model.toDetectPoints(rad_density)
            asdf_points = toData(asdf_points, "numpy")
            pcd = getPointCloud(asdf_points)

            translate = [int(i / row_num) * object_dist[0], 0 * object_dist[1], (i % row_num) * object_dist[2]]

            pcd.translate(translate)
            asdf_pcd_list.append(pcd)

        renderGeometries(asdf_pcd_list, "sample asdf point cloud")
        return True

    @torch.no_grad()
    def step_sample(
        self, sample_num: int, diffuse_steps: int, rad_density: int, category_id: int=0
    ) -> bool:
        self.model.eval()

        object_dist = [2, 0, 2]

        row_num = ceil(sqrt(sample_num))

        print("start diffuse", sample_num, 'asdfs....')
        sampled_array = self.model.sample(
            cond=torch.Tensor([category_id] * sample_num).long().to(self.device),
            batch_seeds=torch.arange(0, sample_num).to(
                self.device
            ),
            diffuse_steps=diffuse_steps,
            step_sample=True
        )

        o3d_viewer = O3DViewer()
        o3d_viewer.createWindow()
        o3d_viewer.update()

        asdf_model = self.toInitialASDFModel()
        for i in range(diffuse_steps + 1):
            print('start create asdf points for diffuse step Itr.' + str(i) + '...')

            o3d_viewer.clearGeometries()

            asdf_pcd_list = []
            for j in tqdm(range(sample_num)):
                asdf_model.loadParams(sampled_array[i][j])
                asdf_points = asdf_model.toDetectPoints(rad_density)
                asdf_points = toData(asdf_points, "numpy")
                pcd = getPointCloud(asdf_points)

                translate = [int(j / row_num) * object_dist[0], 0 * object_dist[1], (j % row_num) * object_dist[2]]

                pcd.translate(translate)
                asdf_pcd_list.append(pcd)

            o3d_viewer.addGeometries(asdf_pcd_list)
            o3d_viewer.update()

        o3d_viewer.run()
        return True
