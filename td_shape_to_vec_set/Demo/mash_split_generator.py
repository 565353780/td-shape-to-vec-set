from td_shape_to_vec_set.Module.mash_split_generator import MashSplitGenerator


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    train_scale = 0.98
    val_scale = 0.01

    mash_split_generator = MashSplitGenerator(dataset_root_folder_path)
    mash_split_generator.convertToSplitFiles(train_scale, val_scale)

    return True
