import torch
import mcubes
import trimesh
import argparse
import numpy as np
from pathlib import Path

from td_shape_to_vec_set.Model.models_ae import kl_d512_m512_l8
from td_shape_to_vec_set.Model.edm_pre_cond import kl_d512_m512_l8_d24_edm


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae-pth', type=str, required=True) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm-pth', type=str, required=True) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    args = parser.parse_args()
    print(args)

    dm = 'kl_d512_m512_l8_d24_edm'
    Path("class_cond_obj/{}".format(dm)).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    ae = kl_d512_m512_l8()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    model = kl_d512_m512_l8_d24_edm()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth)['model'])
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    total = 1000
    iters = 100


    with torch.no_grad():
        for category_id in [18]:
            print(category_id)
            for i in range(1000//iters):
                sampled_array = model.sample(cond=torch.Tensor([category_id]*iters).long().to(device), batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()

                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                for j in range(sampled_array.shape[0]):

                    logits = ae.decode(sampled_array[j:j+1], grid)

                    logits = logits.detach()

                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    verts *= gap
                    verts -= 1

                    m = trimesh.Trimesh(verts, faces)
                    m.export('class_cond_obj/{}/{:02d}-{:05d}.obj'.format(args.dm, category_id, i*iters+j))
