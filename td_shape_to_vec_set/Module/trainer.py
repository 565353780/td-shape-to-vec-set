import os
import sys
import time
import json
import math
import torch
import argparse
import datetime
import numpy as np
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched

from torch.utils.tensorboard import SummaryWriter

from util.datasets import build_shape_surface_occupancy_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_class_cond
import models_ae


class Trainer(object):
    def __init__(self) -> None:
        return

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        ae: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        log_writer=None,
        args=None,
    ):
        model.train(True)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)
        print_freq = 20

        accum_iter = args.accum_iter

        optimizer.zero_grad()

        if log_writer is not None:
            print("log_dir: {}".format(log_writer.log_dir))

        for data_iter_step, (points, labels, surface, categories) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, args
                )

            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            surface = surface.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    _, x = ae.encode(surface)

                loss = criterion(model, x, categories)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_iter_step + 1) % accum_iter == 0,
            )
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            min_lr = 10.0
            max_lr = 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", max_lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, data_loader, model, ae, criterion, device):
        metric_logger = misc.MetricLogger(delimiter="  ")
        header = "Test:"

        # switch to evaluation mode
        model.eval()

        for points, labels, surface, categories in metric_logger.log_every(
            data_loader, 50, header
        ):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            surface = surface.to(device, non_blocking=True)
            categories = categories.to(device, non_blocking=True)
            # compute output

            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    _, x = ae.encode(surface)

                loss = criterion(model, x, categories)

            batch_size = surface.shape[0]

            metric_logger.update(loss=loss.item())

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self):
        parser = argparse.ArgumentParser("Latent Diffusion", add_help=False)
        parser.add_argument(
            "--batch_size",
            default=64,
            type=int,
            help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
        )
        parser.add_argument("--epochs", default=800, type=int)
        parser.add_argument(
            "--accum_iter",
            default=1,
            type=int,
            help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
        )

        # Model parameters
        parser.add_argument(
            "--model",
            default="kl_d512_m512_l8_edm",
            type=str,
            metavar="MODEL",
            help="Name of model to train",
        )

        parser.add_argument(
            "--ae",
            default="kl_d512_m512_l8",
            type=str,
            metavar="MODEL",
            help="Name of autoencoder",
        )

        parser.add_argument("--ae-pth", required=True, help="Autoencoder checkpoint")

        parser.add_argument(
            "--point_cloud_size", default=2048, type=int, help="input size"
        )

        # Optimizer parameters
        parser.add_argument(
            "--clip_grad",
            type=float,
            default=None,
            metavar="NORM",
            help="Clip gradient norm (default: None, no clipping)",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.05,
            help="weight decay (default: 0.05)",
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=None,
            metavar="LR",
            help="learning rate (absolute lr)",
        )
        parser.add_argument(
            "--blr",
            type=float,
            default=1e-4,
            metavar="LR",  # 2e-4
            help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
        )
        parser.add_argument(
            "--layer_decay",
            type=float,
            default=0.75,
            help="layer-wise lr decay from ELECTRA/BEiT",
        )

        parser.add_argument(
            "--min_lr",
            type=float,
            default=1e-6,
            metavar="LR",
            help="lower lr bound for cyclic schedulers that hit 0",
        )

        parser.add_argument(
            "--warmup_epochs",
            type=int,
            default=40,
            metavar="N",
            help="epochs to warmup LR",
        )

        # Dataset parameters
        parser.add_argument(
            "--data_path",
            default="/ibex/ai/home/zhanb0b/data",
            type=str,
            help="dataset path",
        )

        parser.add_argument(
            "--output_dir",
            default="./output/",
            help="path where to save, empty for no saving",
        )
        parser.add_argument(
            "--log_dir", default="./output/", help="path where to tensorboard log"
        )
        parser.add_argument(
            "--device", default="cuda", help="device to use for training / testing"
        )
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--resume", default="", help="resume from checkpoint")

        parser.add_argument(
            "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
        )
        parser.add_argument(
            "--eval", action="store_true", help="Perform evaluation only"
        )
        parser.add_argument(
            "--dist_eval",
            action="store_true",
            default=False,
            help="Enabling distributed evaluation (recommended during training for faster monitor",
        )
        parser.add_argument("--num_workers", default=60, type=int)
        parser.add_argument(
            "--pin_mem",
            action="store_true",
            help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
        )
        parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
        parser.set_defaults(pin_mem=True)

        # distributed training parameters
        parser.add_argument(
            "--world_size", default=1, type=int, help="number of distributed processes"
        )
        parser.add_argument("--local_rank", default=-1, type=int)
        parser.add_argument("--dist_on_itp", action="store_true")
        parser.add_argument(
            "--dist_url",
            default="env://",
            help="url used to set up distributed training",
        )

        args = parser.parse_args()

        misc.init_distributed_mode(args)

        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(", ", ",\n"))

        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset_train = build_shape_surface_occupancy_dataset("train", args=args)
        dataset_val = build_shape_surface_occupancy_dataset("val", args=args)

        if True:  # args.distributed:
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                        "This will slightly alter validation results as extra duplicate entries are added to achieve "
                        "equal num of samples per-process."
                    )
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            # prefetch_factor=2,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            # batch_size=args.batch_size,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        ae = models_ae.__dict__[args.ae]()
        ae.eval()
        print("Loading autoencoder %s" % args.ae_pth)
        ae.load_state_dict(torch.load(args.ae_pth, map_location="cpu")["model"])

        ae.to(device)

        model = models_class_cond.__dict__[args.model]()
        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print("number of params (M): %.2f" % (n_parameters / 1.0e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        # # build optimizer with layer-wise lr decay (lrd)
        # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
        #     layer_decay=args.layer_decay
        # )
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
        loss_scaler = NativeScaler()

        criterion = models_class_cond.__dict__["EDMLoss"]()

        print("criterion = %s" % str(criterion))

        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(
                f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}"
            )
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model,
                ae,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                log_writer=log_writer,
                args=args,
            )
            if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            if epoch % 5 == 0 or epoch + 1 == args.epochs:
                test_stats = evaluate(data_loader_val, model, ae, criterion, device)
                print(
                    f"loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3f}"
                )

                if log_writer is not None:
                    log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

                # if args.output_dir and misc.is_main_process():
                #     if log_writer is not None:
                #         log_writer.flush()
                #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                #         f.write(json.dumps(log_stats) + "\n")

            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(
                    os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
