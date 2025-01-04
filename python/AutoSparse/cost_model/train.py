import os, sys
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.optim import SGD, Adam
import logging
import time
import argparse
from datetime import datetime
from copy import deepcopy
from AutoSparse.utils import logger_init
from AutoSparse.model import cuda_device_id
from AutoSparse.cost_model.config import Config
from AutoSparse.cost_model.model import *
from AutoSparse.cost_model.dataset_loader import *

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


def AccTopK(true_labels: List[torch.Tensor], predicted_labels: List[torch.Tensor], k=5):
    """Get TopK accuracy ratio for ranking.

    Parameters
    ----------
    true_labels : List[torch.Tensor]
        _description_
    predicted_labels : List[torch.Tensor]
        _description_
    k : int, optional
        _description_, by default 5

    Returns
    -------
    (min_true_labels, min_pre_labels, acc_topk)
        _description_
    """
    assert len(true_labels) == len(predicted_labels) and len(true_labels) > 0
    min_true_labels = 0.0
    min_pre_labels = 0.0
    for i in range(len(true_labels)):
        min_true_labels += min(true_labels[i])
        _, indices = torch.topk(predicted_labels[i], k, largest=False, sorted=True)
        min_pre_labels += min(true_labels[i][indices])
    return min_true_labels, min_pre_labels, min_true_labels / min_pre_labels


def SaveModelAndConfig(net, config: Config, losses, filepath: str):
    state_dict = deepcopy(net.state_dict())
    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
            "dataset_dirname_prefixs_lst": config.dataset_dirname_prefixs_lst,
            "batch_size": config.batch_size,
            "model_save_per_epoch": config.model_save_per_epoch,
            "leaning_rate": config.leaning_rate,
            "epoch": config.epoch,
            "is_save_loss_data": config.is_save_loss_data,
            "tensor_name_set": config.batch_size,
            "is_waco_net": config.is_waco_net,
            "is_net_forward1": config.is_net_forward1,
            "in_channels": config.in_channels,
            "D": config.D,
            "middle_channel_num": config.middle_channel_num,
            "token_embedding_size": config.token_embedding_size,
        },
        "losses": losses,
    }
    torch.save(checkpoint, filepath)


def TrainWithWACO(config: Config):
    """Using waco train method, which train a batch schedule data in same sparse matrix."""

    data_flag = "_".join(("_".join(dataset_dirname_prefixs_lst)).split(os.sep))
    model_save_dir = os.path.join(
        root, "python", "AutoSparse", "cost_model", "model_log"
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    device = config.device

    logging.info("##############Define Module##############")
    net = AutoSparseNet(config)
    net = net.to(device)

    loss_func = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(net.parameters(), lr=config.leaning_rate)

    # Get dataset
    logging.info("##############Load Dataset##############")
    spmtx_dataset = LoadSparseMatrixDataset(
        os.path.join(root, "dataset", "train.txt"), batch_size=1, shuffle=True
    )
    spmtx_train_data, spmtx_val_data = spmtx_dataset.load_train_val_data(
        os.path.join(root, "dataset", "train.txt"),
        os.path.join(root, "dataset", "validation.txt"),
    )

    logging.info("############## Start Train ##############")
    train_losses_batch = []
    train_losses_epoch = []
    val_losses_batch = []
    val_losses_epoch = []
    for epoch in range(config.epoch):
        # Train
        net.train()
        start_time = time.time()
        train_loss = 0
        train_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(
            spmtx_train_data
        ):
            torch.cuda.empty_cache()
            shapes = shapes.to(device)
            sparse_matrix = ME.SparseTensor(
                coordinates=coords, features=features, device=device
            )

            num_row, num_col, nnz, coo = (
                get_coo_from_csr_file(  # Notice the coo is [[row, col, 1]]
                    os.path.join(
                        root, "dataset", "total_dataset", mtx_names[0] + ".csr"
                    )
                )
            )
            msg = f"{mtx_names[0]}, Shape{(num_row, num_col, nnz)}"
            logging.info(msg)

            for prefix_idx, dataset_dirname_prefix in enumerate(
                dataset_dirname_prefixs_lst
            ):
                if "sddmm" in dataset_dirname_prefix:
                    mtx_name = mtx_names[0] * 2
                else:
                    mtx_name = mtx_names[0]
                if not os.path.isfile(
                    os.path.join(
                        root, "dataset", dataset_dirname_prefix, mtx_name + ".txt"
                    )
                ):
                    continue
                sche_train_data = LoadScheduleDataset(
                    dataset_dirname_prefix,
                    mtx_name,
                    batch_size=config.batch_size,
                    shuffle=True,
                ).load_data()
                for sche_batchidx, (schedules, relative_runtimes) in enumerate(
                    sche_train_data
                ):
                    if len(schedules) < 2:
                        break
                    relative_runtimes = relative_runtimes.to(device)
                    optimizer.zero_grad()
                    predict = net.forward(schedules, shapes, sparse_matrix)

                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = relative_runtimes[iu[0]], relative_runtimes[iu[1]]
                    sign = (true1 - true2).sign()
                    loss = loss_func(pred1, pred2, sign)
                    train_loss += loss.item()
                    train_loss_cnt += 1
                    train_losses_batch.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    if (
                        sparse_batchidx % 1 == 0
                        and sche_batchidx == 0
                        and prefix_idx == 0
                    ):
                        msg = (
                            f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(spmtx_train_data)}], "
                            f"Sche Batch Count: {len(sche_train_data)}, "
                            f"Train loss : {loss.item():.3f}"
                        )
                        logging.info(msg)

        # Validation
        net.eval()
        min_true_labels_top1, min_pre_labels_top1 = 0.0, 0.0
        min_true_labels_top5, min_pre_labels_top5 = 0.0, 0.0
        with torch.no_grad():
            valid_loss = 0
            valid_loss_cnt = 0
            for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(
                spmtx_val_data
            ):
                torch.cuda.empty_cache()
                shapes = shapes.to(device)
                sparse_matrix = ME.SparseTensor(
                    coordinates=coords, features=features, device=device
                )

                for prefix_idx, dataset_dirname_prefix in enumerate(
                    dataset_dirname_prefixs_lst
                ):
                    if "sddmm" in dataset_dirname_prefix:
                        mtx_name = mtx_names[0] * 2
                    else:
                        mtx_name = mtx_names[0]
                    if not os.path.isfile(
                        os.path.join(
                            root, "dataset", dataset_dirname_prefix, mtx_name + ".txt"
                        )
                    ):
                        continue
                    sche_val_data = LoadScheduleDataset(
                        dataset_dirname_prefix,
                        mtx_name,
                        batch_size=config.batch_size * 2,
                        shuffle=True,
                    ).load_data()
                    for sche_batchidx, (schedules, relative_runtimes) in enumerate(
                        sche_val_data
                    ):
                        if len(schedules) < 5:
                            break
                        relative_runtimes = relative_runtimes.to(device)
                        predict = net.forward(schedules, shapes, sparse_matrix)
                        iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                        pred1, pred2 = predict[iu[0]], predict[iu[1]]
                        true1, true2 = (
                            relative_runtimes[iu[0]],
                            relative_runtimes[iu[1]],
                        )
                        sign = (true1 - true2).sign()
                        loss = loss_func(pred1, pred2, sign)
                        valid_loss += loss.item()
                        valid_loss_cnt += 1
                        val_losses_batch.append(loss.item())
                        (
                            min_true_labels_top1_local,
                            min_pre_labels_top1_local,
                            acc_top1,
                        ) = AccTopK([relative_runtimes], [predict], 1)
                        (
                            min_true_labels_top5_local,
                            min_pre_labels_top5_local,
                            acc_top5,
                        ) = AccTopK([relative_runtimes], [predict], 5)
                        min_true_labels_top1 += min_true_labels_top1_local
                        min_pre_labels_top1 += min_pre_labels_top1_local
                        min_true_labels_top5 += min_true_labels_top5_local
                        min_pre_labels_top5 += min_pre_labels_top5_local

                        if (
                            sparse_batchidx % 1 == 0
                            and sche_batchidx == 0
                            and prefix_idx == 0
                        ):
                            msg = (
                                f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(spmtx_val_data)}], "
                                f"AccTop1: {acc_top1:.3f}, AccTop5: {acc_top5:.3f}, Valid loss: {loss.item():.3f}"
                            )
                            logging.info(msg)

        end_time = time.time()
        train_loss /= train_loss_cnt
        valid_loss /= valid_loss_cnt
        train_losses_batch.append(train_loss)
        val_losses_batch.append(valid_loss)
        acc_top1 = min_true_labels_top1 / min_pre_labels_top1
        acc_top5 = min_true_labels_top5 / min_pre_labels_top5
        msg = (
            f"*********************Epoch Result************************\n"
            f"--- Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"AccTop1: {acc_top1:.3f}, AccTop5: {acc_top5:.3f}, "
            f"Valid loss: {valid_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s\n"
            f"*********************************************"
        )
        logging.info(msg)

        if (epoch + 1) % config.model_save_per_epoch == 0:
            model_name = (
                "autosparse_net_"
                + str(config.is_waco_net)
                + "_"
                + str(config.is_net_forward1)
                + "_"
                + data_flag
                + "_epoch_"
                + str(epoch)
                + datetime.now().strftime("_%Y%m%d_%H%M%S")
                + ".pth"
            )
            SaveModelAndConfig(
                net,
                config,
                {
                    "train_losses_batch": train_losses_batch,
                    "val_losses_batch": val_losses_batch,
                    "train_losses_epoch": train_losses_epoch,
                    "val_losses_epoch": val_losses_epoch,
                },
                os.path.join(model_save_dir, model_name),
            )
            logging.info(f"Save model to {os.path.join(model_save_dir, model_name)}")


def TrainFull(config: Config):
    """
    Mix all the schedule and schedule, and random select a batch data from them to train.
    So a batch data contain different sparse matrix and differt schedule config, which are
    mixed in same batch to train.
    """
    data_flag = "_".join(("_".join(dataset_dirname_prefixs_lst)).split(os.sep))
    model_save_dir = os.path.join(
        root, "python", "AutoSparse", "cost_model", "model_log"
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    log_dir = os.path.join(root, "python", "AutoSparse", "cost_model", "log")
    logger_init("cost_model_train_" + data_flag, log_dir=log_dir)

    device = torch.device(
        "cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu"
    )

    net = AutoSparseNet(
        in_channels=1,
        middle_channels=64,
        embedding_size=128,
        D=2,
        tensor_name_set=None,
    )
    net = net.to(device)


if __name__ == "__main__":
    # create argument parser obj
    parser = argparse.ArgumentParser(
        description="Train the autosparse_net with specified parameters."
    )

    # Add command args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=768,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--epoch", type=int, default=75, help="Number of epochs to train"
    )
    parser.add_argument(
        "--dataset_platform",
        type=str,
        default="xeon_platinum8272cl",
        choices=["epyc_7543", "xeon_platinum8272cl", "full"],
        help="Schedule dataset samples from the cpu platform.",
    )
    parser.add_argument(
        "--dataset_op",
        type=str,
        default="spmm",
        choices=[
            "spmv",
            "spmm",
            "sddmm",
            "spmv_spmm",
            "spmv_spmm_sddmm",
            "spmv_sddmm",
            "spmm_sddmm",
        ],
        help="Schedule dataset samples from the cpu platform.",
    )
    parser.add_argument(
        "--model_save_epoch",
        type=int,
        default=15,
        help="Save model in every some epochs.",
    )
    parser.add_argument(
        "--is_save_loss_data",
        type=int,
        default=1,
        help="Save train and validation loss data in every epoch and batch.",
    )

    parser.add_argument(
        "--is_waco_net",
        type=int,
        default=0,
        help="Is or not use waconet to embed sparse matrix.",
    )
    parser.add_argument(
        "--is_net_forward1",
        type=int,
        default=1,
        help="Which forward func will be used in AutoSparseNet.",
    )
    parser.add_argument(
        "--middle_channel_num",
        type=int,
        default=64,
        help="Middle channel number in conv embedding net.",
    )
    parser.add_argument(
        "--token_embedding_size",
        type=int,
        default=128,
        help="Embedding size for tokens of attention.",
    )

    # Parse args
    args = parser.parse_args()
    config = Config()
    config.batch_size = args.batch_size
    ops = (args.dataset_op).split("_")
    if (
        args.dataset_platform == "epyc_7543"
        or args.dataset_platform == "xeon_platinum8272cl"
    ):
        platforms = [args.dataset_platform]
    else:
        platforms = ["epyc_7543", "xeon_platinum8272cl"]
        logging.info(f"########## Use full platform dataset with {str(platforms)}")
    dataset_dirname_prefixs_lst = []
    for pl in platforms:
        for op in ops:
            dataset_dirname_prefixs_lst.append(pl + "/" + op)
    config.dataset_dirname_prefixs_lst = dataset_dirname_prefixs_lst
    config.model_save_per_epoch = args.model_save_epoch
    config.leaning_rate = args.learning_rate
    config.epoch = args.epoch
    config.is_save_loss_data = args.is_save_loss_data
    config.is_waco_net = args.is_waco_net
    config.is_net_forward1 = args.is_net_forward1
    config.middle_channel_num = args.middle_channel_num
    config.token_embedding_size = args.token_embedding_size

    config.LoggerInit()

    TrainWithWACO(config)


"""

export CUDA_VISIBLE_DEVICES=2
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmv 

export CUDA_VISIBLE_DEVICES=3
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmm 

export CUDA_VISIBLE_DEVICES=4
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmv_spmm_sddmm 

export CUDA_VISIBLE_DEVICES=5
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmm --is_net_forward1 0

export CUDA_VISIBLE_DEVICES=6
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmm --middle_channel_num 256 --is_waco_net 1

export CUDA_VISIBLE_DEVICES=7
python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/train.py --dataset_op spmv_spmm_sddmm --middle_channel_num 256 --is_waco_net 1
"""
