import sys
sys.path.append('../ma-sh/')
sys.path.append('../distribution-manage/')

import os

from td_shape_to_vec_set.Module.mash_trainer import MashTrainer


def demo():
    dataset_root_folder_path = os.environ['HOME'] + "/chLi/Dataset/"
    batch_size = 16
    accum_iter = 16
    num_workers = 16
    model_file_path = None
    # model_file_path = "../../output/20241225_15:15:14/total_model_last.pth".replace('../../', './')
    device = "auto"
    warm_step_num = 2000
    finetune_step_num = -1
    lr = 2e-4
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    trainer = MashTrainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        model_file_path,
        device,
        warm_step_num,
        finetune_step_num,
        lr,
        ema_start_step,
        ema_decay_init,
        ema_decay,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.train()
    return True
