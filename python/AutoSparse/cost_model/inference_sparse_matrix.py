""" Inference SparseMatrixEmbedNet
"""

import os, sys
import torch
import numpy as np
import MinkowskiEngine as ME
import argparse

from AutoSparse.utils import *
from AutoSparse.cost_model.model import *
from AutoSparse.cost_model.config import *
from AutoSparse.cost_model.dataset_loader import *
from AutoSparse.cost_model.utils import *


root = os.getenv("AUTOSPARSE_HOME")


if __name__ == "__main__":
    # create argument parser obj
    parser = argparse.ArgumentParser(
        description="Train the autosparse_net with specified parameters."
    )

    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model pth file name, such as autosparse_net_0_1_xeon_platinum8272cl_spmm_epoch_99_20250124_043656.pth",
    )

    parser.add_argument(
        "--mtx_name_set_filename",
        type=str,
        default="validation.txt",
        help="Sparse matrix name set filename.",
    )

    args = parser.parse_args()

    config = Config()
    config.net_name_prefix = "Inference_SparseMatrixEmbedNet"
    config.LoggerInit()
    net, _ = LoadModelAndConfig(
        AutoSparseNet,
        config,
        os.path.join(
            root,
            "python",
            "AutoSparse",
            "cost_model",
            "model_log",
            args.model_filename
        ),
    )

    device = config.device
    net.to(device)
    net.eval()

    file_path = os.path.join(root, "dataset", args.mtx_name_set_filename)
    sparse_matrix_dataset = SparseMatrixDataset(
        file_path,
        {"std_rows": 1, "std_cols": 1, "std_nnzs": 1, "mean_rows": 0, "mean_cols": 0, "mean_nnzs": 0}
    )
    dataset_iter = DataLoader(
        sparse_matrix_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=SparseMatrixDataset.generate_batch,
    )

    save_dir = os.path.join(root, "dataset", "sparse_matrix_features", args.model_filename[0:-4])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for sparse_batchidx, (mtx_names, coords, features, shapes) in enumerate(dataset_iter):
            print(f"Inference : {sparse_batchidx}/{len(dataset_iter)}")
            if sparse_batchidx % 30 == 0:
                torch.cuda.empty_cache()
            sparse_matrixs = ME.SparseTensor(
                coordinates=coords, features=features, device=device
            )
            feature_vecs = net.embed_sparse_matirx(sparse_matrixs)
            for idx, mtx_name in enumerate(mtx_names):
                SaveSparseMatrixFeatureVec(feature_vecs[idx], mtx_name, save_dir)
