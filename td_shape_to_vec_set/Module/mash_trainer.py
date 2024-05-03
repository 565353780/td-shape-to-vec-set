import os
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from transformers import optimization
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from td_shape_to_vec_set.Loss.edm import EDMLoss
from td_shape_to_vec_set.Dataset.mash import MashDataset
from td_shape_to_vec_set.Dataset.image_embedding import ImageEmbeddingDataset
from td_shape_to_vec_set.Model.edm_pre_cond import EDMPrecond
from td_shape_to_vec_set.Method.path import createFileFolder, renameFile, removeFile
from td_shape_to_vec_set.Method.time import getCurrentTime
from td_shape_to_vec_set.Module.logger import Logger


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class MashTrainer(object):
    def __init__(self):
        self.mash_channel = 400
        self.sh_2d_degree = 3
        self.sh_3d_degree = 2
        self.channels = int(
            6 + (2 * self.sh_2d_degree + 1) + ((self.sh_3d_degree + 1) ** 2)
        )
        self.n_heads = 1
        self.d_head = 768
        self.depth = 12

        self.batch_size = 32
        self.accumulation_steps = 1
        self.num_workers = 16
        self.lr = 1e-4
        self.weight_decay = 1e-10
        self.factor = 0.9
        self.patience = 1000
        self.min_lr = 1e-6
        self.warmup_epochs = 4
        self.train_epochs = 100000
        self.step = 0
        self.eval_step = 0
        self.loss_min = float("inf")
        self.eval_loss_min = float("inf")
        self.log_folder_name = (
            getCurrentTime()
            + "_lr"
            + str(self.lr)
            + "_b"
            + str(self.batch_size * self.accumulation_steps)
            + "_warmup"
            + str(self.warmup_epochs)
            + "_train"
            + str(self.train_epochs)
            + "_mash"
            + str(self.mash_channel)
            + "_sh2d"
            + str(self.sh_2d_degree)
            + "_sh3d"
            + str(self.sh_3d_degree)
            + "_nheads"
            + str(self.n_heads)
            + "_dheah"
            + str(self.d_head)
            + "_depth"
            + str(self.depth)
        )

        HOME = os.environ["HOME"]
        dataset_folder_path_list = [
            HOME + "/Dataset/",
            "/data2/lch/Dataset/",
        ]
        for dataset_folder_path in dataset_folder_path_list:
            if not os.path.exists(dataset_folder_path):
                continue

            self.dataset_root_folder_path = dataset_folder_path
            break

        dist.init_process_group(backend="nccl")

        self.device_id = dist.get_rank() % torch.cuda.device_count()
        self.device = "cuda:" + str(self.device_id)

        self.model = EDMPrecond(
            n_latents=self.mash_channel,
            channels=self.channels,
            n_heads=self.n_heads,
            d_head=self.d_head,
            depth=self.depth,
        ).to(self.device)
        self.model = DDP(self.model, device_ids=[self.device_id])

        mash_dataset = MashDataset(self.dataset_root_folder_path)
        image_embedding_dataset = ImageEmbeddingDataset(self.dataset_root_folder_path)

        mash_sampler = DistributedSampler(mash_dataset)
        image_embedding_sampler = DistributedSampler(image_embedding_dataset)

        self.mash_dataloader = DataLoader(
            dataset=mash_dataset,
            sampler=mash_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

        self.image_embedding_dataloader = DataLoader(
            dataset=image_embedding_dataset,
            sampler=image_embedding_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

        lr_scale = (
            self.batch_size * self.accumulation_steps * dist.get_world_size() / 256
        )
        self.lr *= lr_scale
        self.min_lr *= lr_scale
        self.optimizer = AdamW(
            self.model.module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optimization.get_polynomial_decay_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(
                self.warmup_epochs
                * len(self.mash_dataloader)
                / self.accumulation_steps
            ),
            num_training_steps=int(
                self.train_epochs * len(self.mash_dataloader) / self.accumulation_steps
            ),
            lr_end=self.min_lr,
            power=3,
        )

        self.logger = Logger()

        self.loss_func = EDMLoss()
        return

    def loadSummaryWriter(self):
        if dist.get_rank() != 0:
            return False

        self.logger.setLogFolder("./logs/" + self.log_folder_name + "/")
        return True

    def loadModel(self, model_file_path, resume_model_only=False):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        self.model.module.load_state_dict(model_dict["model"])

        if not resume_model_only:
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print(
            "\t load model success! start training from step " + str(self.step) + "..."
        )
        return True

    def saveModel(self, save_model_file_path):
        if dist.get_rank() != 0:
            return False

        model_dict = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "eval_step": self.eval_step,
            "loss_min": self.loss_min,
            "eval_loss_min": self.eval_loss_min,
            "log_folder_name": self.log_folder_name,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def getLr(self) -> float:
        return self.optimizer.state_dict()["param_groups"][0]["lr"]

    def trainStep(self, mash_params, condition):
        self.model.train()

        loss = self.loss_func(self.model, mash_params, condition)

        loss_item = loss.clone().detach().cpu().numpy()

        self.logger.addScalar("Train/loss", loss_item, self.step)

        if loss_item < self.loss_min:
            self.loss_min = loss_item
            self.saveModel("./output/" + self.log_folder_name + "/model_best.pth")

        loss = loss / self.accumulation_steps
        loss.backward()

        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()

            self.model.zero_grad()
            self.optimizer.zero_grad()
        return loss_item

    def train(self):
        print_progress = dist.get_rank() == 0

        if not self.logger.isValid():
            self.loadSummaryWriter()

        total_epoch = 10000000

        self.model.zero_grad()
        for epoch in range(total_epoch):
            if print_progress:
                print("[INFO][Trainer::train]")
                print(
                    "\t start training, epoch : "
                    + str(epoch + 1)
                    + "/"
                    + str(total_epoch)
                    + "..."
                )
            if print_progress:
                pbar = tqdm(total=len(self.mash_dataloader))
            for data in self.mash_dataloader:
                self.step += 1

                mash_params = data["mash_params"].to(self.device, non_blocking=True)
                categories = data["category_id"].to(self.device, non_blocking=True)
                loss = self.trainStep(mash_params, categories)

                if print_progress:
                    pbar.set_description(
                        "[Mash] LOSS %.6f LR %.4f*1e-6" % (loss, self.getLr() * 1e6)
                    )
                    pbar.update(1)

                self.logger.addScalar("Lr/lr", self.getLr(), self.step)

                if self.step % self.accumulation_steps == 0:
                    self.scheduler.step()

            if print_progress:
                pbar.close()

            self.saveModel("./output/" + self.log_folder_name + "/model_last.pth")

            if print_progress:
                pbar = tqdm(total=len(self.image_embedding_dataloader))
            for data in self.image_embedding_dataloader:
                self.step += 1

                mash_params = data["mash_params"].to(self.device, non_blocking=True)
                image_embedding = data["image_embedding"]
                key_idx = np.random.choice(len(image_embedding.keys()))
                key = list(image_embedding.keys())[key_idx]
                image_embedding = image_embedding[key].to(self.device, non_blocking=True)

                loss = self.trainStep(mash_params, image_embedding)

                if print_progress:
                    pbar.set_description(
                        "[Image Embedding] LOSS %.6f LR %.4f*1e-6" % (loss, self.getLr() * 1e6)
                    )
                    pbar.update(1)

                self.logger.addScalar("Lr/lr", self.getLr(), self.step)

                if self.step % self.accumulation_steps == 0:
                    self.scheduler.step()

            if print_progress:
                pbar.close()

            self.saveModel("./output/" + self.log_folder_name + "/model_last.pth")

        return True
