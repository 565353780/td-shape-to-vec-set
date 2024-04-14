import os
import numpy as np
from math import ceil
from typing import Tuple


class MashSplitGenerator(object):
    def __init__(self, dataset_root_folder_path: str) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_folder_path = self.dataset_root_folder_path + "Mash/"
        self.split_folder_path = self.dataset_root_folder_path + "Split/"

        assert os.path.exists(self.mash_folder_path)

        return

    def toCategoryRelFilePathList(self, dataset_name: str, category_name: str) -> list:
        mash_category_folder_path = (
            self.mash_folder_path + dataset_name + "/mash/" + category_name + "/"
        )

        rel_file_path_list = []

        print("[INFO][MashSplitGenerator::toCategoryRelFilePathList]")
        print("\t start search npy files...")
        print("\t mash_category_folder_path:", mash_category_folder_path)
        for root, _, files in os.walk(mash_category_folder_path):
            rel_folder_path = root.split(mash_category_folder_path)[1] + "/"

            for file_name in files:
                if file_name[-4:] != ".npy":
                    continue

                mash_file_path = mash_category_folder_path + rel_folder_path + file_name

                if not os.path.exists(mash_file_path):
                    continue

                rel_file_path = rel_folder_path + file_name
                rel_file_path_list.append(rel_file_path)

        return rel_file_path_list

    def convertToCategorySplits(
        self,
        dataset_name: str,
        category_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> Tuple[list, list, list]:
        if not os.path.exists(self.dataset_root_folder_path):
            print("[ERROR][MashSplitGenerator::convertToCategorySplits]")
            print("\t dataset root folder not exist!")
            print("\t dataset_root_folder_path:", self.dataset_root_folder_path)
            return [], [], []

        rel_file_path_list = self.toCategoryRelFilePathList(dataset_name, category_name)

        permut_rel_file_path_list = np.random.permutation(rel_file_path_list)

        rel_file_path_num = len(rel_file_path_list)

        if rel_file_path_num < 3:
            print("[WARN][MashSplitGenerator::convertToCategorySplits]")
            print("\t category shape num < 3!")
            print("\t rel_file_path_num:", rel_file_path_num)
            return [], [], []

        train_split_num = ceil(train_scale * rel_file_path_num)
        val_split_num = ceil(val_scale * rel_file_path_num)

        if rel_file_path_num == 3:
            train_split_num = 1
            val_split_num = 1

        if train_split_num + val_split_num == rel_file_path_num:
            train_split_num -= 1

        train_split = permut_rel_file_path_list[:train_split_num]
        val_split = permut_rel_file_path_list[
            train_split_num : train_split_num + val_split_num
        ]
        test_split = permut_rel_file_path_list[train_split_num + val_split_num :]

        return train_split.tolist(), val_split.tolist(), test_split.tolist()

    def convertToCategorySplitFiles(
        self,
        dataset_name: str,
        category_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        train_split, val_split, test_split = self.convertToCategorySplits(
            dataset_name, category_name, train_scale, val_scale
        )

        if len(train_split) + len(val_split) + len(test_split) == 0:
            print("[ERROR][MashSplitGenerator::convertToCategorySplitFiles]")
            print("\t convertToCategorySplits failed!")
            return False

        save_split_folder_path = (
            self.split_folder_path + "mash/" + dataset_name + "/" + category_name + "/"
        )

        os.makedirs(save_split_folder_path, exist_ok=True)

        with open(save_split_folder_path + "train.txt", "w") as f:
            for train_name in train_split:
                f.write(train_name + "\n")

        with open(save_split_folder_path + "val.txt", "w") as f:
            for val_name in val_split:
                f.write(val_name + "\n")

        with open(save_split_folder_path + "test.txt", "w") as f:
            for test_name in test_split:
                f.write(test_name + "\n")

        return True

    def convertToDatasetSplitFiles(
        self,
        dataset_name: str,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        categories = os.listdir(self.mash_folder_path + dataset_name + "/mash/")

        for i, category in enumerate(categories):
            print("[INFO][MashSplitGenerator::convertToDatasetSplitFiles]")
            print(
                "\t start generate mash dataset split: "
                + dataset_name
                + "["
                + category
                + "], "
                + str(i + 1)
                + "/"
                + str(len(categories))
                + "..."
            )

            self.convertToCategorySplitFiles(
                dataset_name, category, train_scale, val_scale
            )

        return True

    def convertToSplitFiles(
        self,
        train_scale: float = 0.8,
        val_scale: float = 0.1,
    ) -> bool:
        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            self.convertToDatasetSplitFiles(dataset_name, train_scale, val_scale)

        return True
