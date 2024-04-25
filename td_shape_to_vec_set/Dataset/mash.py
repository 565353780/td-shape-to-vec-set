import os
import torch
import numpy as np
from torch.utils.data import Dataset

from td_shape_to_vec_set.Config.shapenet import CATEGORY_IDS


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_folder_path = self.dataset_root_folder_path + "Mash/"

        assert os.path.exists(self.mash_folder_path)

        self.paths_list = []

        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)

            for i, category in enumerate(categories):
                class_folder_path = dataset_folder_path + category + "/"

                mash_filename_list = os.listdir(class_folder_path)

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(i + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for mash_filename in mash_filename_list:
                    mash_file_path = class_folder_path + mash_filename

                    if not os.path.exists(mash_file_path):
                        continue

                    self.paths_list.append([mash_file_path, CATEGORY_IDS[category]])
        return

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        mash_file_path, category_id = self.paths_list[index]

        mash_params = np.load(mash_file_path, allow_pickle=True).item()

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        mash_params = np.hstack([rotate_vectors, positions, mask_params, sh_params])
        mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        feed_dict = {
            "mash_params": torch.tensor(mash_params).float(),
            "category_id": category_id,
        }

        return feed_dict
