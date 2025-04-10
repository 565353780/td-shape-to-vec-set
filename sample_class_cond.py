import os
import torch
import mcubes
import trimesh
import numpy as np
from tqdm import trange

from td_shape_to_vec_set.Model.models_ae import kl_d512_m512_l8
from td_shape_to_vec_set.Model.edm_pre_cond import kl_d512_m512_l8_d24_edm
from td_shape_to_vec_set.Method.path import createFileFolder


if __name__ == "__main__":
    ae_pth  = '/home/chli/chLi/Model/3DShape2VecSet/ae/kl_d512_m512_l8/checkpoint-199.pth'
    dm_pth  = '/home/chli/chLi/Model/3DShape2VecSet/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth'

    total = 20

    ae_device = torch.device('cpu')
    model_device = torch.device('cuda:0')

    dm = 'kl_d512_m512_l8_d24_edm'
    save_result_folder_path = '/home/chli/chLi/Results/3DShape2VecSet/output/gen_class_cond/' + dm + '/'
    os.makedirs(save_result_folder_path, exist_ok=True)

    ae = kl_d512_m512_l8()
    ae.eval()
    ae.load_state_dict(torch.load(ae_pth, weights_only=False)['model'])
    ae.to(ae_device)

    model = kl_d512_m512_l8_d24_edm()
    model.eval()

    model.load_state_dict(torch.load(dm_pth, weights_only=False)['model'])
    model.to(model_device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(ae_device, non_blocking=True)

    valid_category_id_list = [
        '02691156', # 0: airplane
        '02773838', # 2: bag
        '02828884', # 6: bench
        '02876657', # 9: bottle
        '02958343', # 16: bottle
        '03001627', # 18: chair
        '03211117', # 22: monitor
        '03261776', # 23: earphone
        '03325088', # 24: spigot
        '03467517', # 26: guitar
        '03513137', # 27: helmet
        '03636649', # 30: lamp
        '03710193', # 33: mailbox
        '03948459', # 40: gun
        '04090263', # 44: long-gun
        '04225987', # 46: skateboard
        '04256520', # 47: sofa
        '04379243', # 49: table
        '04468005', # 52: train
        '04530566', # 53: watercraft
    ]
    valid_category_id_list = [0, 2, 6, 9, 16, 18, 22, 23, 24, 26, 27, 39, 33, 49, 44, 46, 47, 49, 52, 53]

    valid_category_id_list = [
        18,
    ]

    with torch.no_grad():
        for category_id in valid_category_id_list:
            full_sample_array_list = model.sample(cond=torch.Tensor([category_id]*total).long().to(model_device), batch_seeds=torch.arange(0, total).to(model_device))

            for denoise_step, sample_array in enumerate(full_sample_array_list):
                sampled_array = sample_array.float().to(ae_device)

                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                print('start save shapes at denoising step', denoise_step, '...')
                for shape_id in trange(sampled_array.shape[0]):

                    logits = ae.decode(sampled_array[shape_id:shape_id+1], grid)

                    logits = logits.detach()

                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    verts *= gap
                    verts -= 1

                    m = trimesh.Trimesh(verts, faces)
                    current_save_shape_path = save_result_folder_path + \
                        'category_' + str(category_id) + '/' + \
                        'shape_' + str(shape_id) + '/' + \
                        'denoise_' + str(denoise_step) + '.obj'
                    createFileFolder(current_save_shape_path)
                    m.export(current_save_shape_path)
