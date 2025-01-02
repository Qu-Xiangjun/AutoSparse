import os, sys
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.optim import SGD, Adam
import logging
import time
from copy import deepcopy
from AutoSparse.utils import logger_init
from AutoSparse.model import cuda_device_id

from model import *
from dataset_loader import *

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


def TrainWithWACO(
    dataset_dirname_prefixs_lst: List[str], batch_size=64, model_save_per_epoch=20
):
    """Using waco train method, which train a batch schedule data in same sparse matrix.
    Parameters
        ----------
        dataset_dirname_prefixs_lst : List[str]
            Schedule dataset dirname, such as ['epyc_7543/spmm', 'epyc_7543/spmv']
        batch_size : int
        model_save_per_epoch : int
    """

    data_flag = "_".join(("_".join(dataset_dirname_prefixs_lst)).split(os.sep))
    model_save_dir = os.path.join(
        root, "python", "AutoSparse", "cost_model", "model_log"
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    log_dir = os.path.join(root, "python", "AutoSparse", "cost_model", "log")
    logger_init("cost_model_train", log_dir=log_dir)

    device = torch.device(
        "cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu"
    )

    logging.info("##############Define Module##############")
    net = AutoSparseNet(
        in_channels=1,
        middle_channels=128,
        embedding_size=128,
        D=2,
        tensor_name_set=None,
    )
    net = net.to(device)

    loss_func = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(net.parameters(), lr=1e-4)

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
    for epoch in range(100):
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

            for prefix_idx, dataset_dirname_prefix in enumerate(
                dataset_dirname_prefixs_lst
            ):
                sche_train_data = LoadScheduleDataset(
                    dataset_dirname_prefix,
                    mtx_names[0],
                    batch_size=batch_size,
                    shuffle=True,
                ).load_data()
                for sche_batchidx, (schedules, runtimes) in enumerate(sche_train_data):
                    if len(schedules) < 2:
                        break
                    runtimes = runtimes.to(device)
                    optimizer.zero_grad()
                    predict = net.forward(schedules, shapes, sparse_matrix)

                    iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                    pred1, pred2 = predict[iu[0]], predict[iu[1]]
                    true1, true2 = runtimes[iu[0]], runtimes[iu[1]]
                    sign = (true1 - true2).sign()
                    loss = loss_func(pred1, pred2, sign)
                    train_loss += loss.item()
                    train_loss_cnt += 1

                    loss.backward()
                    optimizer.step()
                    if (
                        sparse_batchidx % 5 == 0
                        and sche_batchidx == 0
                        and prefix_idx == 0
                    ):
                        msg = f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(spmtx_train_data)}], Train loss : {loss.item():.3f}"
                        logging.info(msg)

        # Validation
        net.eval()
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
                    sche_train_data = LoadScheduleDataset(
                        dataset_dirname_prefix,
                        mtx_names,
                        batch_size=batch_size * 2,
                        shuffle=True,
                    )
                    for sche_batchidx, (schedules, runtimes) in enumerate(
                        sche_train_data
                    ):
                        if len(schedules) < 5:
                            break
                        schedules, runtimes = schedules.to(device), runtimes.to(device)
                        predict = net.forward(schedules, shapes, sparse_matrix)
                        iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                        pred1, pred2 = predict[iu[0]], predict[iu[1]]
                        true1, true2 = runtimes[iu[0]], runtimes[iu[1]]
                        sign = (true1 - true2).sign()
                        loss = loss_func(pred1, pred2, sign)
                        valid_loss += loss.item()
                        valid_loss_cnt += 1
                        if (
                            (sparse_batchidx + 1) % 100 == 0
                            and (sche_batchidx + 1) == 0
                            and (prefix_idx + 1) == 0
                        ):
                            msg = f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(spmtx_train_data)}], Valid loss : {loss.item():.3f}"
                            logging.info(msg)

        end_time = time.time()
        train_loss /= train_loss_cnt
        valid_loss /= valid_loss_cnt
        msg = f"""*********************Epoch Result************************
            --- Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s
            *********************************************"""
        logging.info(msg)

        if (epoch + 1) % model_save_per_epoch == 0:
            state_dict = deepcopy(net.state_dict())
            model_name = "autosparse_net" + data_flag + "_epoch" + str(epoch) + ".pth"
            torch.save(state_dict, os.path.join(model_save_dir, model_name))
            logging.info(f"Save model to {os.path.join(model_save_dir, model_name)}")


def TrainFull(
    dataset_dirname_prefixs_lst: List[str], batch_size=128, model_save_per_epoch=20
):
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
    logger_init("cost_model_train", log_dir=log_dir)

    device = torch.device(
        "cuda:" + str(cuda_device_id) if torch.cuda.is_available() else "cpu"
    )

    net = AutoSparseNet(
        in_channels=1,
        middle_channels=128,
        embedding_size=128,
        D=2,
        tensor_name_set=None,
    )
    net = net.to(device)


if __name__ == "__main__":
    TrainWithWACO(["epyc_7543/spmv"])
