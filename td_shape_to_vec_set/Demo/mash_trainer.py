def demo():
    from td_shape_to_vec_set.Module.mash_trainer import MashTrainer

    model_file_path = "./output/pretrain-S/model_last.pth"

    mash_trainer = MashTrainer()
    #mash_trainer.loadModel(model_file_path, True)
    mash_trainer.train()
    return True
