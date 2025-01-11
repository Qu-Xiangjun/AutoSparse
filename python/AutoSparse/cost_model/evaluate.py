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


def AccTopK(
    true_labels: List[torch.Tensor],
    predicted_labels: List[torch.Tensor],
    k=5,
    largest=False,
):
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
    (min_true_labels, min_pre_labels, score)
        min_true_labels: The min lable of all true labels set.
        min_pre_labels: The true lable of min pre labels in pre labels set.
        score: The topk score in the sets.
    """
    assert len(true_labels) == len(predicted_labels) and len(true_labels) > 0
    if largest:
        func = max
    else:
        func = min
    min_true_labels = 0.0
    min_pre_labels = 0.0
    for i in range(len(true_labels)):
        min_true_labels += func(true_labels[i])
        _, indices = torch.topk(predicted_labels[i], k, largest=largest, sorted=True)
        min_pre_labels += func(true_labels[i][indices])
    if largest:
        score = min_pre_labels / min_true_labels
    else:
        score = min_true_labels / min_pre_labels
    return min_true_labels, min_pre_labels, score


def EvaluateHelp(
    device,
    net: nn.Module,
    batch_size,
    spmtx_val_data,
    dataset_dirname_prefixs_lst,
    loss_func,
    epoch,
    data_handle_method,
):
    net.eval()

    val_losses_batch = []
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
                    batch_size=batch_size * 2,
                    shuffle=True,
                    handle_method="relative_max",
                ).load_data()
                for sche_batchidx, (schedules, relative_runtimes) in enumerate(
                    sche_val_data
                ):
                    schedules = schedules[
                        : int(len(schedules) / len(dataset_dirname_prefixs_lst))
                    ]
                    relative_runtimes = relative_runtimes[
                        : int(len(relative_runtimes) / len(dataset_dirname_prefixs_lst))
                    ]
                    if len(schedules) < 5:
                        break
                    relative_runtimes = relative_runtimes.to(device)
                    predict = net.forward(schedules, shapes, sparse_matrix)

                    if isinstance(loss_func, nn.MarginRankingLoss):
                        iu = torch.triu_indices(predict.shape[0], predict.shape[0], 1)
                        pred1, pred2 = predict[iu[0]], predict[iu[1]]
                        true1, true2 = (
                            relative_runtimes[iu[0]],
                            relative_runtimes[iu[1]],
                        )
                        sign = (true1 - true2).sign()
                        loss = loss_func(pred1, pred2, sign)
                    elif isinstance(loss_func, LambdaRankingLoss):
                        loss = loss_func(predict, relative_runtimes)

                    valid_loss += loss.item()
                    valid_loss_cnt += 1
                    val_losses_batch.append(loss.item())
                    (
                        min_true_labels_top1_local,
                        min_pre_labels_top1_local,
                        acc_top1,
                    ) = AccTopK(
                        [relative_runtimes],
                        [predict],
                        1,
                        data_handle_method == "relative_max",
                    )
                    (
                        min_true_labels_top5_local,
                        min_pre_labels_top5_local,
                        acc_top5,
                    ) = AccTopK(
                        [relative_runtimes],
                        [predict],
                        5,
                        data_handle_method == "relative_max",
                    )
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

    return (
        val_losses_batch,
        valid_loss,
        valid_loss_cnt,
        min_true_labels_top1,
        min_pre_labels_top1,
        min_true_labels_top5,
        min_pre_labels_top5,
    )
