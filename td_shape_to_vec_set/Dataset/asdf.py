import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from td_shape_to_vec_set.Config.shapenet import CATEGORY_IDS


class ASDFDataset(Dataset):
    def __init__(self, asdf_dataset_folder_path: str) -> None:
        self.asdf_file_list = []
        # self.context_files_list = []

        # """
        self.asdf_file_list = [
            "/home/chli/Nutstore Files/paper-materials-ASDF/Dataset/final.npy"
        ] * 1204
        return
        # """

        self.loadFinalASDFDataset(asdf_dataset_folder_path)
        return
        self.loadDataset(asdf_dataset_folder_path)
        return

    def loadFinalASDFDataset(self, final_asdf_dataset_folder_path: str) -> bool:
        final_asdf_filename_list = os.listdir(final_asdf_dataset_folder_path)
        for final_asdf_filename in final_asdf_filename_list:
            if final_asdf_filename[-10:] != "_final.npy":
                continue

            asdf_file_path = final_asdf_dataset_folder_path + final_asdf_filename

            self.asdf_file_list.append(asdf_file_path)

        return True

    def loadDataset(self, asdf_dataset_folder_path: str) -> bool:
        class_foldername_list = os.listdir(asdf_dataset_folder_path)

        for class_foldername in class_foldername_list:
            model_folder_path = asdf_dataset_folder_path + class_foldername + "/"
            if not os.path.exists(model_folder_path):
                continue

            model_filename_list = os.listdir(model_folder_path)

            for model_filename in tqdm(model_filename_list):
                asdf_folder_path = model_folder_path + model_filename + "/"
                if not os.path.exists(asdf_folder_path):
                    continue

                asdf_filename_list = os.listdir(asdf_folder_path)

                if "final.npy" not in asdf_filename_list:
                    continue

                context_files = []

                for asdf_filename in asdf_filename_list:
                    if asdf_filename == "final.npy":
                        continue

                    if asdf_filename[-4:] != ".npy":
                        continue

                    context_files.append(asdf_folder_path + asdf_filename)

                self.asdf_file_list.append(asdf_folder_path + "final.npy")
                # self.context_files_list.append(context_files)

        return True

    def __len__(self):
        return len(self.asdf_file_list)

    def __getitem__(self, idx):
        asdf_file_path = self.asdf_file_list[idx]
        asdf = np.load(asdf_file_path, allow_pickle=True).item()["params"]
        shuffle_asdf = np.random.permutation(asdf)

        """
        context_file_path = choice(self.context_files_list[idx])
        context = (
            np.load(context_file_path, allow_pickle=True)
            .item()["params"]
            .reshape(1, 100, 40)
        )
        """

        return (
            torch.from_numpy(shuffle_asdf).type(torch.float32),
            CATEGORY_IDS["02691156"],
        )

        positions = shuffle_asdf[:, :6]
        params = shuffle_asdf[:, 6:]

        embedding_positions = ((positions + 1.0) * 128.0).astype(np.longlong)

        return (
            torch.from_numpy(embedding_positions).type(torch.long),
            torch.from_numpy(params).type(torch.float32),
            CATEGORY_IDS["02691156"],
        )
