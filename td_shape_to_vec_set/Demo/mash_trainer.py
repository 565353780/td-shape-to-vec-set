def demo():
    from td_shape_to_vec_set.Module.mash_trainer import MashTrainer

    model_file_path = "./output/20240414_12:01:15_lr0.0001_b128_warmup4_train100000_mash40_sh2d4_sh3d4_nheads8_dheah64_depth24/model_last.pth"
    print_progress = True

    mash_trainer = MashTrainer()
    mash_trainer.loadModel(model_file_path, True)
    mash_trainer.train(print_progress)
    return True
