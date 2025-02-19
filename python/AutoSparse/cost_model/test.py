import os, sys
import torch
import numpy as np
import MinkowskiEngine as ME

from AutoSparse.cost_model.model import AutoSparseNet
from AutoSparse.cost_model.model import *
from AutoSparse.cost_model.config import *
from AutoSparse.cost_model.utils import *
from AutoSparse.cost_model.evaluate import *
from AutoSparse.cost_model.waco_net import WACONet, SparseMatrixDataset, collate_fn, SuperScheduleDataset

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


def test_autosparse_net(net, config: Config):
    dataset_dirname_prefixs_lst = ['xeon_platinum8272cl/spmv', 'xeon_platinum8272cl/spmm', 'xeon_platinum8272cl/sddmm']

    if config.loss_fn == "MarginRankingLoss":
        loss_func = nn.MarginRankingLoss(margin=1)
    elif config.loss_fn == "LambdaRankingLoss":
        loss_func = LambdaRankingLoss(config.device)

    device = config.device
    net.to(device)

    # Get dataset
    logging.info("##############Load Dataset##############")
    spmtx_dataset = LoadSparseMatrixDataset(
        os.path.join(root, "dataset", "train.txt"), batch_size=1, shuffle=True
    )
    spmtx_train_data, spmtx_val_data = spmtx_dataset.load_train_val_data(
        os.path.join(root, "dataset", "train_demo.txt"),
        os.path.join(root, "dataset", "validation_demo.txt"),
    )

    logging.info("############## Start Test ##############")
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
                    batch_size=config.batch_size * 2,
                    shuffle=True,
                    handle_method=config.data_handle_method,
                ).load_data()
                for sche_batchidx, (schedules, relative_runtimes) in enumerate(
                    sche_val_data
                ):
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
                        config.data_handle_method == "relative_max",
                    )
                    (
                        min_true_labels_top5_local,
                        min_pre_labels_top5_local,
                        acc_top5,
                    ) = AccTopK(
                        [relative_runtimes],
                        [predict],
                        5,
                        config.data_handle_method == "relative_max",
                    )
                    min_true_labels_top1 += min_true_labels_top1_local
                    min_pre_labels_top1 += min_pre_labels_top1_local
                    min_true_labels_top5 += min_true_labels_top5_local
                    min_pre_labels_top5 += min_pre_labels_top5_local

                    if (
                        sparse_batchidx % 1 == 0
                        and sche_batchidx == 0
                    ):
                        msg = (
                            f"{dataset_dirname_prefix}, Matrix: {mtx_names[0]}, Matrix Batch[{sparse_batchidx}/{len(spmtx_val_data)}], "
                            f"AccTop1: {acc_top1:.3f}, AccTop5: {acc_top5:.3f}, Valid loss: {loss.item():.3f}"
                        )
                        logging.info(msg)

        valid_loss /= valid_loss_cnt
        if config.data_handle_method == "relative_min":
            acc_top1 = min_true_labels_top1 / min_pre_labels_top1
            acc_top5 = min_true_labels_top5 / min_pre_labels_top5
        elif config.data_handle_method == "relative_max":
            acc_top1 = min_pre_labels_top1 / min_true_labels_top1
            acc_top5 = min_pre_labels_top5 / min_true_labels_top5
        else:
            assert False

        msg = (
            f"*********************Epoch Result************************\n"
            f"Valid AccTop1: {acc_top1:.3f}, Valid AccTop5: {acc_top5:.3f}, "
            f"Valid loss: {valid_loss:.3f}\n"
            f"*********************************************"
        )
        logging.info(msg)


def test_waco_net(net, config):
    dataset_dirname_prefixs_lst = ['xeon_platinum8272cl/spmv', 'xeon_platinum8272cl/spmm', 'xeon_platinum8272cl/sddmm']

    loss_func = nn.MarginRankingLoss(margin=1)

    device = config.device
    net.to(device)

    # Get dataset
    logging.info("##############Load Dataset##############")
    SparseMatrix_Dataset = SparseMatrixDataset(
        os.path.join(root, "dataset", "validation_demo.txt")
    )
    valid_SparseMatrix = torch.utils.data.DataLoader(
        SparseMatrix_Dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    logging.info("############## Start Test ##############")
    net.eval()
    val_losses_batch = []
    min_true_labels_top1, min_pre_labels_top1 = 0.0, 0.0
    min_true_labels_top5, min_pre_labels_top5 = 0.0, 0.0
    with torch.no_grad():
        valid_loss = 0
        valid_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(
            valid_SparseMatrix
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
                
                if 'spmm' in dataset_dirname_prefix:
                    op_type = 'SpMM'
                elif 'spmv' in dataset_dirname_prefix:
                    op_type = 'SpMV'
                elif 'sddmm' in dataset_dirname_prefix:
                    op_type = 'SDDMM'
                else:
                    assert False
                
                SuperSchedule_Dataset = SuperScheduleDataset(
                    dataset_dirname_prefix,
                    mtx_name,
                    handle_method=config.data_handle_method,
                )
                train_SuperSchedule = torch.utils.data.DataLoader(
                    SuperSchedule_Dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=0,
                )

                for sche_batchidx, (schedules, relative_runtimes) in enumerate(
                    train_SuperSchedule
                ):
                    if len(schedules) < 5:
                        break
                    relative_runtimes = relative_runtimes.to(device)
                    schedules = schedules.to(device)
                    query_feature = net.embed_sparse_matrix(sparse_matrix, shapes)
                    query_feature = query_feature.expand(
                        (schedules.shape[0], query_feature.shape[1])
                    )
                    predict = net.forward_after_query(query_feature, schedules, op_type)
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
                    ) = AccTopK(
                        [relative_runtimes],
                        [predict],
                        1,
                        config.data_handle_method == "relative_max",
                    )
                    (
                        min_true_labels_top5_local,
                        min_pre_labels_top5_local,
                        acc_top5,
                    ) = AccTopK(
                        [relative_runtimes],
                        [predict],
                        5,
                        config.data_handle_method == "relative_max",
                    )
                    min_true_labels_top1 += min_true_labels_top1_local
                    min_pre_labels_top1 += min_pre_labels_top1_local
                    min_true_labels_top5 += min_true_labels_top5_local
                    min_pre_labels_top5 += min_pre_labels_top5_local

                    if (
                        sparse_batchidx % 1 == 0
                        and sche_batchidx == 0
                    ):
                        msg = (
                            f"{dataset_dirname_prefix}, Matrix: {mtx_names[0]}, Matrix Batch[{sparse_batchidx}/{len(SparseMatrix_Dataset)}], "
                            f"AccTop1: {acc_top1:.3f}, AccTop5: {acc_top5:.3f}, Valid loss: {loss.item():.3f}"
                        )
                        logging.info(msg)

        valid_loss /= valid_loss_cnt
        if config.data_handle_method == "relative_min":
            acc_top1 = min_true_labels_top1 / min_pre_labels_top1
            acc_top5 = min_true_labels_top5 / min_pre_labels_top5
        elif config.data_handle_method == "relative_max":
            acc_top1 = min_pre_labels_top1 / min_true_labels_top1
            acc_top5 = min_pre_labels_top5 / min_true_labels_top5
        else:
            assert False

        msg = (
            f"*********************Epoch Result************************\n"
            f"Valid AccTop1: {acc_top1:.3f}, Valid AccTop5: {acc_top5:.3f}, "
            f"Valid loss: {valid_loss:.3f}\n"
            f"*********************************************"
        )
        logging.info(msg)


if __name__ == "__main__":
    config = Config()
    config.net_name_prefix = "Test_AutoSparseNet"
    config.LoggerInit()
    net, _ = LoadModelAndConfig(AutoSparseNet, config, os.path.join(
            root,
            "python",
            "AutoSparse",
            "cost_model",
            "model_log",
            "autosparse_net_0_1_xeon_platinum8272cl_spmm_epoch_99_20250124_043656.pth",
        ))
    test_autosparse_net(net, config)

    waco_net = WACONet(1, 1, 2)
    net, _ = LoadModelAndConfig(waco_net, config, os.path.join(
            root,
            "python",
            "AutoSparse",
            "cost_model",
            "waco_model_log",
            "waco_net_xeon_platinum8272cl_spmm_epoch_99_20250217_114237.pth",
        ))
    config.net_name_prefix = "Test_WACONet"
    test_waco_net(waco_net, config)
