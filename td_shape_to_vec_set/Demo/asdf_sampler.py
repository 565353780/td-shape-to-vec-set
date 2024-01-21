import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def demo():
    from td_shape_to_vec_set.Module.asdf_sampler import ASDFSampler

    model_file_path = "./output/v1/model_last.pth"
    device = "cuda"

    sample_num = 4
    diffuse_steps = 36
    rad_density = 5
    category_id = 0

    asdf_sampler = ASDFSampler(model_file_path, device)
    asdf_sampler.step_sample(sample_num, diffuse_steps, rad_density, category_id)
    return True
