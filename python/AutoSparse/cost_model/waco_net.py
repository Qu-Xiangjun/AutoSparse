import os, sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import MinkowskiEngine as ME
import torch.utils
import torch.utils.data
from AutoSparse.utils import get_coo_from_csr_file
from AutoSparse.cost_model.tokenizer import Tokenizer
from torch.optim import SGD, Adam
from AutoSparse.cost_model.config import Config, logger_init
from AutoSparse.cost_model.utils import SaveModelAndConfig
from AutoSparse.cost_model.evaluate import AccTopK
import logging
import time
from datetime import datetime

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


# ------------------------------------------------------------------------------
# ---------------------------- sparse matrix loader ----------------------------
# ------------------------------------------------------------------------------


def collate_fn(list_data):
    coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
    )

    mtxnames_batch = [d["mtxname"] for d in list_data]
    shapes_batch = torch.stack([d["shape"] for d in list_data])

    return mtxnames_batch, coords_batch, features_batch, shapes_batch


class SparseMatrixDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        with open(filepath) as f:
            self.names = f.read().splitlines()
        # Preparing Data 统计训练数据中一些数据信息，用于标准化后面的shape和nnz信息
        # 理解这里即使是构建测试集数据Dataset，也会和训练数据使用一样的标准信息，符合预期
        self.standardize = {}
        self.normalize = {}
        with open(os.path.join(root, "dataset", "train.txt")) as f:
            total_rows, total_cols, total_nnzs = [], [], []
            for filename in f.read().splitlines():
                csr = np.fromfile(
                    root + "/dataset/total_dataset/" + filename + ".csr",
                    count=3,
                    dtype="<i4",
                )
                total_rows.append(csr[0])
                total_cols.append(csr[1])
                total_nnzs.append(csr[2])
            self.standardize["mean_rows"] = np.mean(total_rows)
            self.standardize["mean_cols"] = np.mean(total_cols)
            self.standardize["mean_nnzs"] = np.mean(total_nnzs)
            self.standardize["std_rows"] = np.std(total_rows)
            self.standardize["std_cols"] = np.std(total_cols)
            self.standardize["std_nnzs"] = np.std(total_nnzs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        filename = self.names[idx]
        num_row, num_col, nnz, coo = get_coo_from_csr_file(
            os.path.join(root, "dataset", "total_dataset", filename + ".csr")
        )

        # standardize
        num_row = (num_row - self.standardize["mean_rows"]) / self.standardize[
            "std_rows"
        ]
        num_col = (num_col - self.standardize["mean_cols"]) / self.standardize[
            "std_cols"
        ]
        nnz = (nnz - self.standardize["mean_nnzs"]) / self.standardize["std_nnzs"]

        # To ME Sparse Tensor
        coordinates = torch.from_numpy(coo[:, :2]).to(torch.int32)
        features = torch.ones((len(coo), 1)).to(torch.float32)
        label = torch.tensor([[0]]).to(torch.float32)
        shape = torch.tensor([num_row, num_col, nnz]).to(torch.float32)

        return {
            "mtxname": filename,
            "coordinates": coordinates,
            "features": features,
            "label": label,
            "shape": shape,
        }


# ------------------------------------------------------------------------------
# ---------------------------- superschedule loader ----------------------------
# ------------------------------------------------------------------------------


class SuperScheduleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dirname_prefix, mtx_name, handle_method="relative_max"):
        file_path = os.path.join(
            root, "dataset", dataset_dirname_prefix, mtx_name + ".txt"
        )
        if not os.path.isfile(file_path):
            logging.INFO(f"Find file don't exist {file_path}")
            return
        with open(file_path) as f:
            lines = f.read().splitlines()

        split_ = [0] + [1 << p for p in range(17)]
        index_ = ["i1", "i0", "k1", "k0", "j1", "j0", "None"]
        format_ = [0, 1, 2, 3, 4]  # (C,U)
        parnum_ = [i for i in range(300)]
        parchunk_ = [1 << p for p in range(13)]  # [1,4096]

        schedules = []
        runtimes = []

        max_runtime = 0.0
        min_runtime = 10000
        for line in lines:
            runtime = float(line.split(" ")[-1])
            if runtime < 1000:
                max_runtime = max(max_runtime, runtime)
                min_runtime = min(min_runtime, runtime)

        for idx, line in enumerate(lines):
            runtime = float(line.split(" ")[-1])
            if runtime >= 1000:
                continue

            embeded_vec = []
            parsed_sch = Tokenizer.ScheduleParse(line)
            for primitive_vec in parsed_sch:
                primitive_name = primitive_vec[0]
                if (
                    primitive_name == Tokenizer.PRIMITIVES[0]
                    or primitive_name == Tokenizer.PRIMITIVES[3]
                ):  # ["{f/l}split" i i0 256 i1 16]
                    embeded_vec.append(split_.index(int(primitive_vec[-1])))
                elif (
                    primitive_name == Tokenizer.PRIMITIVES[2]
                ):  # ["fmode" A i0 1 i1 1 k0 1 k1 1]
                    for i in range(2, len(primitive_vec), 2):
                        embeded_vec.append(format_.index(int(primitive_vec[i + 1])))
                elif (
                    primitive_name == Tokenizer.PRIMITIVES[4]
                ):  # ["lreorder" j0 i0 i1 k0 k1 j1]
                    orders = primitive_vec[1:]
                    perm = np.zeros((len(index_), len(index_)))
                    for order_item_idx, order_item in enumerate(orders):
                        perm[index_.index(order_item), order_item_idx] = 1
                    perm = perm.flatten()
                    embeded_vec.append(perm)
                elif (
                    primitive_name == Tokenizer.PRIMITIVES[5]
                    or primitive_name == Tokenizer.PRIMITIVES[7]
                ):  # ["lvector" None], ["lparallel" j0]
                    embeded_vec.append(index_.index(primitive_vec[1]))
                elif primitive_name == Tokenizer.PRIMITIVES[6]:  # ["lunroll" k1, 4]
                    embeded_vec.append(index_.index(primitive_vec[1]))
                elif (
                    primitive_name == Tokenizer.PRIMITIVES[8]
                ):  # ["openmp_parameter" 128 2]
                    embeded_vec.append(parnum_.index(int(primitive_vec[1])))
                    embeded_vec.append(parchunk_.index(int(primitive_vec[2])))

            concat = np.array(embeded_vec, dtype=object)
            concat = np.hstack(concat)

            if handle_method == "relative_min":
                runtime = torch.tensor(runtime / max_runtime)
            elif handle_method == "relative_max":
                runtime = torch.tensor(min_runtime / runtime)
            else:
                assert False
            schedules.append(concat)
            runtimes.append(runtime)

        schedules = np.stack(schedules, axis=0)
        runtimes = np.stack(runtimes, axis=0)
        self.schedules = schedules.astype(np.float32)
        self.runtimes = runtimes.astype(np.float32)

        # To TorchTensor
        self.schedules = torch.from_numpy(self.schedules)
        self.runtimes = torch.from_numpy(self.runtimes)

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        return self.schedules[idx], self.runtimes[idx]


# ------------------------------------------------------------------------------
# ----------------------------------- model ------------------------------------
# ------------------------------------------------------------------------------


class WACONet(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 32
    PLANES = (16, 32, 64, 64)

    def __init__(self, in_channels, out_channels, D=2):
        nn.Module.__init__(self)
        self.D = D

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        # Sparse Matrix Query
        self.inplanes = self.INIT_DIM
        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=5, stride=1, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer8 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer9 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer10 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer11 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer12 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer13 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )
        self.layer14 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiReLU(inplace=True),
        )

        self.glob_pool = nn.Sequential(
            ME.MinkowskiGlobalAvgPooling(), ME.MinkowskiToFeature()
        )

        self.feature = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.matrix_embedding = nn.Sequential(
            nn.Linear(self.INIT_DIM * 14 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Super Schedule
        embeding_cnt = 13
        self.isplit = nn.Embedding(18, 32)
        self.jsplit = nn.Embedding(18, 32)
        self.ksplit = nn.Embedding(18, 32)

        self.formati1 = nn.Embedding(5, 32)
        self.formati0 = nn.Embedding(5, 32)
        self.formatk1 = nn.Embedding(5, 32)
        self.formatk0 = nn.Embedding(5, 32)
        self.paridx = nn.Embedding(7, 32)
        self.parvecidx = nn.Embedding(7, 32)
        self.unrollidx = nn.Embedding(7, 32)
        self.parnum = nn.Embedding(300, 32)
        self.parchunk = nn.Embedding(13, 32)  # For OpenTuner
        self.order = nn.Linear(49, 32)  # 7x7 Permutation

        self.schedule_embedding = nn.Sequential(
            nn.Linear(32 * embeding_cnt, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Final Layer
        self.final = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def embed_sparse_matrix(self, x1: ME.SparseTensor, x2):
        # Sparse Matrix
        y1 = self.layer1(x1)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7 = self.layer7(y6)
        y8 = self.layer8(y7)
        y9 = self.layer9(y8)
        y10 = self.layer10(y9)
        y11 = self.layer11(y10)
        y12 = self.layer12(y11)
        y13 = self.layer13(y12)
        y14 = self.layer14(y13)

        y1 = self.glob_pool(y1)
        y2 = self.glob_pool(y2)
        y3 = self.glob_pool(y3)
        y4 = self.glob_pool(y4)
        y5 = self.glob_pool(y5)
        y6 = self.glob_pool(y6)
        y7 = self.glob_pool(y7)
        y8 = self.glob_pool(y8)
        y9 = self.glob_pool(y9)
        y10 = self.glob_pool(y10)
        y11 = self.glob_pool(y11)
        y12 = self.glob_pool(y12)
        y13 = self.glob_pool(y13)
        y14 = self.glob_pool(y14)

        # y = F.normalize(torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14), dim=1))
        y = torch.cat(
            (y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14), dim=1
        )

        x2 = self.feature(x2[:, :3])
        x1x2 = torch.cat((y, x2), dim=1)
        x1x2 = self.matrix_embedding(x1x2)

        # x1x2 = F.normalize(x1x2)

        return x1x2

    def embed_super_schedule(self, y, op_type):
        # Super Schedule
        sch_feature_vec = []
        sch_feature_vec.append(self.isplit(y[:, 0].long()))
        if "SpMV" == op_type:
            constant_tmp = torch.tensor(np.array([6])).long().expand(y.shape[0]).to(y.device)
            sch_feature_vec.append(self.jsplit(y[:, 1].long()))
            sch_feature_vec.append(self.ksplit(constant_tmp))
            sch_feature_vec.append(self.formati1(y[:, 2].long()))
            sch_feature_vec.append(self.formati0(y[:, 3].long()))
            sch_feature_vec.append(self.formatk1(y[:, 4].long()))
            sch_feature_vec.append(self.formatk0(y[:, 5].long()))
            next_idx = 6
        elif "SpMM" == op_type or "SDDMM" == op_type:
            sch_feature_vec.append(self.ksplit(y[:, 1].long()))
            sch_feature_vec.append(self.formati1(y[:, 2].long()))
            sch_feature_vec.append(self.formati0(y[:, 3].long()))
            sch_feature_vec.append(self.formatk1(y[:, 4].long()))
            sch_feature_vec.append(self.formatk0(y[:, 5].long()))
            sch_feature_vec.append(self.jsplit(y[:, 6].long()))
            next_idx = 7
        else:
            assert False
        if op_type == "SDDMM":
            next_idx = next_idx + 4
        sch_feature_vec.append(self.order(y[:, next_idx : next_idx + 7 * 7]))
        sch_feature_vec.append(self.paridx(y[:, next_idx + 7 * 7].long()))
        sch_feature_vec.append(self.unrollidx(y[:, next_idx + 7 * 7 + 1].long()))
        sch_feature_vec.append(self.parvecidx(y[:, next_idx + 7 * 7 + 2].long()))
        sch_feature_vec.append(self.parnum(y[:, next_idx + 7 * 7 + 3].long()))
        sch_feature_vec.append(self.parchunk(y[:, next_idx + 7 * 7 + 4].long()))
        y = torch.cat(sch_feature_vec, dim=1)
        y = self.schedule_embedding(y)

        # y = F.normalize(y)
        return y

    def forward_after_query(self, x, y, op_type):
        y = self.embed_super_schedule(y, op_type)
        xy = torch.cat((x, y), dim=1)
        xy = self.final(xy)
        return xy.squeeze()

    def forward(self, x, y, op_type):
        return self.forward_after_query(x, y, op_type)


class ResNet14(WACONet):
    LAYERS = (1, 1, 1, 1)


# ------------------------------------------------------------------------------
# ----------------------------- train & validation -----------------------------
# ------------------------------------------------------------------------------


def Train(config: Config):
    cuda_device_id = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Number of available GPUs:", num_gpus)

        device = torch.device("cuda:" + str(cuda_device_id))
        print("Using GPU device:", device)
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")
    config.device = device

    config.model_save_per_epoch = 10
    config.middle_channel_num = 32
    config.loss_fn = "MarginRankingLoss"
    task_cnt = 0
    task_name = ""
    if "spmm" in str(config.dataset_dirname_prefixs_lst):
        task_name += "SpMM"
        task_cnt += 1
    if "spmv" in str(config.dataset_dirname_prefixs_lst):
        task_name += "SpMV"
        task_cnt += 1
    if "sddmm" in str(config.dataset_dirname_prefixs_lst):
        task_name += "SDDMM"
        task_cnt += 1
    msg = {
        "dataset_dirname_prefixs_lst": config.dataset_dirname_prefixs_lst,
        "batch_size": config.batch_size,
        "model_save_per_epoch": config.model_save_per_epoch,
        "leaning_rate": config.leaning_rate,
        "epoch": config.epoch,
        "is_save_loss_data": config.is_save_loss_data,
        "in_channels": config.in_channels,
        "D": config.D,
        "middle_channel_num": config.middle_channel_num,
        "loss_fn": config.loss_fn,
        "data_handle_method": config.data_handle_method,
    }
    for k, v in msg.items():
        print(k, " : ", str(v))

    dataset_dirname_prefixs_lst = config.dataset_dirname_prefixs_lst
    data_flag = "_".join(("_".join(dataset_dirname_prefixs_lst)).split(os.sep))
    model_save_dir = os.path.join(
        root, "python", "AutoSparse", "cost_model", "waco_model_log"
    )
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    logging.info("##############Define Module##############")
    net = WACONet(1, out_channels=1, D=2)
    net = net.to(device)

    loss_func = nn.MarginRankingLoss(margin=1)
    optimizer = Adam(net.parameters(), lr=1e-4)

    logging.info("##############Load Matrix Dataset##############")
    SparseMatrix_Dataset = SparseMatrixDataset(
        os.path.join(root, "dataset", "train.txt")
    )
    train_SparseMatrix = torch.utils.data.DataLoader(
        SparseMatrix_Dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    SparseMatrix_Dataset = SparseMatrixDataset(
        os.path.join(root, "dataset", "validation.txt")
    )
    valid_SparseMatrix = torch.utils.data.DataLoader(
        SparseMatrix_Dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    logging.info("############## Start Train ##############")
    train_losses_batch = []
    train_losses_epoch = []
    val_losses_batch = []
    val_losses_epoch = []
    for epoch in range(config.epoch):
        # train
        net.train()
        start_time = time.time()
        train_loss = 0
        train_loss_cnt = 0
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(
            train_SparseMatrix
        ):
            num_row, num_col, nnz, coo = (
                get_coo_from_csr_file(  # Notice the coo is [[row, col, 1]]
                    os.path.join(
                        root, "dataset", "total_dataset", mtx_names[0] + ".csr"
                    )
                )
            )
            msg = f"{mtx_names[0]}, Shape{(num_row, num_col, nnz)}"
            logging.info(msg)

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
                mtx_filepath = os.path.join(
                        root, "dataset", dataset_dirname_prefix, mtx_name + ".txt"
                    )
                if not os.path.isfile(mtx_filepath):
                    logging.WARNING("Mtx file don't exist: " + mtx_filepath)
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
                    schedules = schedules[
                        : int(len(schedules) / len(dataset_dirname_prefixs_lst))
                    ]
                    relative_runtimes = relative_runtimes[
                        : int(len(relative_runtimes) / len(dataset_dirname_prefixs_lst))
                    ]
                    if len(schedules) < 2:
                        break
                    schedules = schedules.to(device)
                    relative_runtimes = relative_runtimes.to(device)
                    optimizer.zero_grad()
                    query_feature = net.embed_sparse_matrix(sparse_matrix, shapes)
                    query_feature = query_feature.expand(
                        (schedules.shape[0], query_feature.shape[1])
                    )
                    predict = net.forward_after_query(query_feature, schedules, op_type)
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
                            f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(train_SparseMatrix)}], "
                            f"Sche Batch Count: {len(train_SuperSchedule)}, "
                            f"Train loss : {loss.item():.3f}"
                        )
                        logging.info(msg)

        # validation
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
                        schedules = schedules[
                            : int(len(schedules) / len(dataset_dirname_prefixs_lst))
                        ]
                        relative_runtimes = relative_runtimes[
                            : int(
                                len(relative_runtimes)
                                / len(dataset_dirname_prefixs_lst)
                            )
                        ]
                        if len(schedules) <= 5:
                            break
                        schedules = schedules.to(device)
                        relative_runtimes = relative_runtimes.to(device)
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
                            True,
                        )
                        (
                            min_true_labels_top5_local,
                            min_pre_labels_top5_local,
                            acc_top5,
                        ) = AccTopK(
                            [relative_runtimes],
                            [predict],
                            5,
                            True,
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
                                f"Epoch: {epoch}, Matrix Batch[{sparse_batchidx}/{len(valid_SparseMatrix)}], "
                                f"AccTop1: {acc_top1:.3f}, AccTop5: {acc_top5:.3f}, Valid loss: {loss.item():.3f}"
                            )
                            logging.info(msg)

        end_time = time.time()
        train_loss /= train_loss_cnt
        valid_loss /= valid_loss_cnt
        train_losses_epoch.append(train_loss)
        val_losses_epoch.append(valid_loss)
        global_acc_top1 = min_pre_labels_top1 / min_true_labels_top1
        global_acc_top5 = min_pre_labels_top5 / min_true_labels_top5

        msg = (
            f"*********************Epoch Result************************\n"
            f"--- Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"Valid AccTop1: {global_acc_top1:.3f}, Valid AccTop5: {global_acc_top5:.3f}, "
            f"Valid loss: {valid_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s\n"
            f"*********************************************"
        )
        logging.info(msg)

        if (epoch + 1) % config.model_save_per_epoch == 0:
            model_name = (
                "waco_net_"
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


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the waco_net with specified parameters."
    )

    # Add command args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=768,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Number of epochs to train"
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
        default=5,
        help="Save model in every some epochs.",
    )

    # Parse args
    args = parser.parse_args()
    config = Config()
    config.net_name_prefix = "WACONet"
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
    config.epoch = args.epoch
    config.LoggerInit()

    Train(config)

"""

export CUDA_VISIBLE_DEVICES=2
nohup python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/waco_net.py --dataset_op spmv >/dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/waco_net.py --dataset_op spmm >/dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/waco_net.py --dataset_op spmv_spmm_sddmm >/dev/null 2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python $AUTOSPARSE_HOME/python/AutoSparse/cost_model/waco_net.py --dataset_op sddmm  >/dev/null 2>&1 &

"""
