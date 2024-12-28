import sys
sys.path.append("../ma-sh/")
sys.path.append('../distribution-manage/')

import os
import open3d as o3d
from tqdm import tqdm
from math import sqrt, ceil

from td_shape_to_vec_set.Method.time import getCurrentTime
from td_shape_to_vec_set.Module.mash_sampler import MashSampler


def demo():
    model_file_path = "../../output/shapenet_03001627_v1/total_model_last.pth".replace('../.', '')
    transformer_id = 'ShapeNet_03001627'
    use_ema = True
    device = "cpu"

    sample_num = 9
    condition = 18
    diffuse_steps = 18

    timestamp = getCurrentTime()
    save_folder_path = './output/sample/' + timestamp + '/'
    os.makedirs(save_folder_path + '/pcd/', exist_ok=True)


    print(model_file_path)
    mash_sampler = MashSampler(model_file_path, use_ema, device, transformer_id)

    print("start diffuse", sample_num, "mashs....")
    # sampled_array = mash_sampler.sample(sample_num, condition, diffuse_steps)[-1]

    mash_file_path_list = [
        '/home/chli/github/ASDF/ma-sh/output/combined_mash.npy',
    ]
    sampled_array = mash_sampler.sampleWithFixedAnchors(mash_file_path_list, sample_num, condition, diffuse_steps)[-1]

    object_dist = [2, 0, 2]

    row_num = ceil(sqrt(sample_num))

    mash_pcd_list = []

    mash_model = mash_sampler.toInitialMashModel()

    for i in tqdm(range(sample_num)):
        mash_params = sampled_array[i]

        mash_params = mash_sampler.transformer.inverse_transform(mash_params)

        sh2d = 2 * mash_sampler.mask_degree + 1
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

        mash_model.saveParamsFile(save_folder_path + 'mash/sample_' + str(i) + '.npy')

        mash_pcd = mash_model.toSamplePcd()

        if False:
            translate = [
                int(i / row_num) * object_dist[0],
                0 * object_dist[1],
                (i % row_num) * object_dist[2],
            ]

            mash_pcd.translate(translate)

        mash_pcd_list.append(mash_pcd)

    for i in range(len(mash_pcd_list)):
        o3d.io.write_point_cloud(
            save_folder_path + "pcd/sample_pcd_" + str(i) + ".ply",
            mash_pcd_list[i],
            write_ascii=True,
        )
    return True
