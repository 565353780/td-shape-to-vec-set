import sys

sys.path.append("../data-convert/")
sys.path.append("../spherical-harmonics/")
sys.path.append("../a-sdf/")

def demo():
    from td_shape_to_vec_set.Module.asdf_trainer import ASDFTrainer

    model_file_path = './output/v3/model_last.pth'
    print_progress = True

    asdf_trainer = ASDFTrainer()
    # asdf_trainer.loadSummaryWriter()
    asdf_trainer.loadModel(model_file_path, True)
    asdf_trainer.train(print_progress)
    return True
