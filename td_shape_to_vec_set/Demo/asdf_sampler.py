import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")


def demo():
    from td_shape_to_vec_set.Module.asdf_sampler import ASDFSampler

    model_file_path = "./output/v1/model_best.pth"
    device = "cuda"

    sample_num = 10
    rad_density = 5

    asdf_sampler = ASDFSampler(model_file_path, device)
    asdf_sampler.sample(sample_num, rad_density)
    return True
