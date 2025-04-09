import os
import torch
import mcubes
import trimesh
import argparse
import numpy as np
import open3d as o3d

from td_shape_to_vec_set.Model.models_ae import kl_d512_m512_l8


if __name__ == "__main__":
    pcd_file_path = '/home/chli/chLi/Dataset/Thingi10K/mesh_pcd/61258.ply'
    if not os.path.exists(pcd_file_path):
        print('pcd file not exist!')
        exit()

    file_basename = pcd_file_path.split('/')[-1].split('.')[0]

    parser = argparse.ArgumentParser('', add_help=False)
    ae_pth = '/home/chli/chLi/Model/3DShape2VecSet/ae/kl_d512_m512_l8/checkpoint-199.pth'

    dm = 'kl_d512_m512_l8_d24_edm'
    save_result_folder_path = './output/auto_encoder/' + dm + '/'
    os.makedirs(save_result_folder_path, exist_ok=True)

    ae_device = torch.device('cpu')

    pcd = o3d.io.read_point_cloud(pcd_file_path)

    sample_pcd = pcd.farthest_point_down_sample(2048)

    pts = np.asarray(sample_pcd.points)

    pc = torch.from_numpy(pts).to(ae_device, dtype=torch.float32).unsqueeze(0)

    ae = kl_d512_m512_l8()
    ae.eval()
    ae.load_state_dict(torch.load(ae_pth, weights_only=False)['model'])
    ae.to(ae_device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(ae_device, non_blocking=True)

    with torch.no_grad():
        kl, vec_set = ae.encode(pc)

        logits = ae.decode(vec_set, grid)

        logits = logits.detach()

        volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)

        verts *= gap
        verts -= 1

        m = trimesh.Trimesh(verts, faces)
        os.makedirs(save_result_folder_path, exist_ok=True)
        m.export(save_result_folder_path + 'Thingi10K-' + file_basename + '.obj')
