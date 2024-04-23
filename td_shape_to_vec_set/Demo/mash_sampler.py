import sys

sys.path.append("../ma-sh/")


def demo():
    from td_shape_to_vec_set.Module.mash_sampler import MashSampler

    model_file_path = "./output/v2/model_best.pth"
    model_file_path = "/Users/fufu/github/ASDF/tmp/output/20240419_01:53:43_lr1e-05_b1000_warmup800_train100000_mash40_sh2d4_sh3d4_nheads8_dheah64_depth24/model_best.pth"
    device = "cpu"

    sample_num = 9
    diffuse_steps = 36
    category_id = 18

    mash_sampler = MashSampler(model_file_path, device)
    mash_sampler.sample(sample_num, diffuse_steps, category_id)
    return True
