import os, sys
import torch
import numpy as np
import pandas
import MinkowskiEngine as ME
from typing import *
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
from data_helper import process_cache
from AutoSparse.utils import get_coo_from_csr_file

file_dir = os.path.dirname(os.path.abspath(__file__))
root = os.getenv("AUTOSPARSE_HOME")


class SparseMatrixDataset(Dataset):
	def __init__(self, filepath: str, standardize: Dict, normalize: Dict):
		"""_summary_

		Parameters
		----------
		filepath : str
			Sparse matrix names txt file
		standardize : Dict
			Sparse matirx standardized shape info in train dataset.
		normalize : Dict
			Sparse matirx normalized shape info in train dataset.
		"""
		with open(filepath) as f:
			self.names = f.read().splitlines()
		self.standardize = standardize
		self.normalize = normalize

	def __len__(self):
		return len(self.names)

	def __getitem__(self, index):
		filename = self.names[index]
		num_row, num_col, nnz, coo = get_coo_from_csr_file( # Notice the coo is [[row, col, 1]]
			os.path.join(root, "dataset", "total_dataset", filename + '.csr')
		)
		# standardize
		sparsity = nnz / num_row / num_col
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
		shape = torch.tensor([num_row, num_col, nnz, sparsity]).to(torch.float32)

		return {
			"mtxname": filename,
			"coordinates": coordinates,
			"features": features,
			"label": label,
			"shape": shape,
		}

	@staticmethod
	def generate_batch(data_batch):
		coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
			[d["coordinates"] for d in data_batch],
			[d["features"] for d in data_batch],
			[d["label"] for d in data_batch],
		)

		mtxnames_batch = [d["mtxname"] for d in data_batch]
		shapes_batch = torch.stack([d["shape"] for d in data_batch])

		return mtxnames_batch, coords_batch, features_batch, shapes_batch


class LoadSparseMatrixDataset:
	"""Get sparse matrix from dataset by the filename txt file, and nomalize the data info."""

	def __init__(self, train_file_path=None, batch_size=1, shuffle=True):
		"""_summary_

		Parameters
		----------
		train_file_path : _type_, optional
			Sparse matrix names txt file, by default None
		batch_size : int, optional
			_description_, by default 1
		shuffle : bool, optional
			_description_, by default True
		"""
		logging.info(f"### Build sparse matrix dataset.")
		if train_file_path == None:
			train_file_path = os.path.join(root, "dataset", "total.txt")
		# Make statistics of shape info for all train dataset matrix.
		self.standardize = {}
		self.normalize = {}
		logging.info(f"### Make statistics of shape info for sparse matrix dataset.")
		with open(train_file_path) as f:
			total_rows, total_cols, total_nnzs = [], [], []
			for filename in tqdm(
				f.read().splitlines(),
				ncols=110,
				desc="Statistics of sparse matrix dataset",
			):
				csr = np.fromfile(
					os.path.join(root, "dataset", "total_dataset", filename + ".csr"),
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

		assert (
			batch_size == 1
		), "Only support batch_size 1 for load sparse matrix in waco train code."
		self.batch_size = batch_size
		self.shuffle = shuffle

	def load_train_val_data(self, train_file_path=None, val_file_path=None):
		if train_file_path == None:
			train_file_path = os.path.join(root, "dataset", "train.txt")
		if val_file_path == None:
			val_file_path = os.path.join(root, "dataset", "validation.txt")
		train_data = SparseMatrixDataset(
			train_file_path, self.standardize, self.normalize
		)
		val_data = SparseMatrixDataset(val_file_path, self.standardize, self.normalize)
		train_iter = DataLoader(
			train_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		val_iter = DataLoader(
			val_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		return train_iter, val_iter

	def load_test_data(self, test_file_path=None):
		if test_file_path == None:
			test_file_path = os.path.join(root, "dataset", "test.txt")
		test_data = SparseMatrixDataset(
			test_file_path, self.standardize, self.normalize
		)
		test_iter = DataLoader(
			test_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		return test_iter

	def generate_batch(self, data_batch):
		return SparseMatrixDataset.generate_batch(data_batch)


class LoadScheduleDataset(torch.utils.data.Dataset):
	"""
	In a specifical sparse matrix situation, get scheudle of target operator from txt file.
	File name is specific by matrix name.
	"""

	def __init__(self, dataset_dirname_prefix, mtx_name, batch_size=64, shuffle=True):
		"""_summary_

		Parameters
		----------
		dataset_dirname_prefix : str
			Schedule and label dataset fold name.
		mtx_name : str
			Sparse matrix name, such as auz12.
		batch_size : int, optional
			_description_, by default 64
		shuffle : bool, optional
			_description_, by default True
		"""
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.data = []
		file_path = os.path.join(root, "dataset", dataset_dirname_prefix, mtx_name + ".txt")
		if not os.path.isfile(file_path):
			logging.INFO(f"Find file don't exist {file_path}")
			return
		with open(file_path) as f:
			lines = f.read().splitlines()
		max_runtime = 0.0
		for line in lines:
			runtime = float(line.split(" ")[-1])
			if (runtime < 1000):
				max_runtime = max(max_runtime, runtime)
		for line in lines:
			runtime = float(line.split(" ")[-1])
			if (runtime < 1000) :
				runtime = torch.tensor(runtime / max_runtime)
				self.data.append((line, runtime))
		if len(self.data) == 0: # Dataloader will raise error when get num_samples==0.
			self.data = [(" 0.000", 0.000)]

	def load_data(self):
		return DataLoader(
			self.data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)

	def generate_batch(self, data_batch):
		batch_schedule, batch_label = [], []
		for sche, label in data_batch:
			batch_schedule.append(sche)
			batch_label.append(label)
		batch_label = torch.tensor(batch_label)
		return batch_schedule, batch_label


class LoadHybridSparseMatrixScheduleDataSet(torch.utils.data.Dataset):
	"""Load full dataset from sparse matrix file and their's schedule txt file. The
	dataset can contain three different data dimension, which are operators, matrix
	and schedules. All data runtime label will preprocess to relative ranking, so
	that train method can use ranking loss in any shuffled batch data.
	"""

	def __init__(self, train_file_path=None, batch_size=64, shuffle=True):
		logging.info(f"### Build hybric sparse matrix and scheule dataset.")
		if train_file_path == None:
			train_file_path = os.path.join(root, "dataset", "total.txt")
		# Make statistics of shape info for all train dataset matrix.
		self.standardize = {}
		self.normalize = {}
		logging.info(f"### Make statistics of shape info for sparse matrix dataset.")
		with open(train_file_path) as f:
			total_rows, total_cols, total_nnzs = [], [], []
			for filename in tqdm(
				f.read().splitlines(),
				ncols=110,
				desc="Statistics of sparse matrix dataset",
			):
				csr = np.fromfile(
					os.path.join(root, "dataset", "total_dataset", filename + ".csr"),
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

		self.batch_size = batch_size
		self.shuffle = shuffle

	@process_cache()
	def data_process(self, file_path, dataset_dirname_prefixs_lst):
		with open(file_path) as f:
			mtx_names = f.read().splitlines()
		logging.info(
			f"### Convert dataset {file_path} with {len(mtx_names)} matrix and all \
					 schedules in {str(dataset_dirname_prefixs_lst)} to standardized data."
		)
		data = []
		for mtx_name in tqdm(mtx_names, ncols=110, desc="Load sparse matrix with it's schedule: "):
			num_row, num_col, nnz, coo = get_coo_from_csr_file(
				os.path.join(root, "dataset", "total_dataset", mtx_name + '.csr')
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
			coordinates = torch.from_numpy(coo[:, 2]).to(torch.int32)
			features = torch.ones((len(coo), 1)).to(torch.float32)
			shape = torch.tensor([num_row, num_col, nnz]).to(torch.float32)

			# load schedule data
			for schedule_dataset_dir_prefix in dataset_dirname_prefixs_lst:
				with open(
					os.path.join(
						root, "dataset", schedule_dataset_dir_prefix, mtx_name + ".txt"
					)
				) as f:
					sche_lines = f.read().splitlines()
				max_runtime = 0.0
				for sche in sche_lines:
					runtime = float(sche.split(" ")[-1])
					if (runtime < 1000):
						max_runtime = max(max_runtime, runtime)
				for sche in sche_lines:
					handled_runtime = float(sche.split(" ")[-1]) / max_runtime
					if (runtime < 1000):
						handled_runtime = torch.tensor(handled_runtime)
					data.append(
						{
							"coordinates": coordinates,
							"features": features,
							"shape": shape,
							"schedule": sche,
							"label": handled_runtime,
						}
					)
		return data

	def load_train_val_data(
		self, train_file_path=None, val_file_path=None, dataset_dirname_prefixs_lst=None
	):
		if train_file_path == None:
			train_file_path = os.path.join(root, "dataset", "train.txt")
		if val_file_path == None:
			val_file_path = os.path.join(root, "dataset", "validation.txt")
		if dataset_dirname_prefixs_lst == None or dataset_dirname_prefixs_lst == []:
			dataset_dirname_prefixs_lst = [os.path.join("epyc_7543", "spmv")]
		train_data = self.data_process(train_file_path, dataset_dirname_prefixs_lst)
		val_data = self.data_process(val_file_path, dataset_dirname_prefixs_lst)
		train_iter = DataLoader(
			train_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		val_iter = DataLoader(
			val_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		return train_iter, val_iter

	def load_test_data(self, test_file_path=None, dataset_dirname_prefixs_lst=None):
		if test_file_path == None:
			test_file_path = os.path.join(root, "dataset", "test.txt")
		if dataset_dirname_prefixs_lst == None or dataset_dirname_prefixs_lst == []:
			dataset_dirname_prefixs_lst = [os.path.join("epyc_7543", "spmv")]
		test_data = self.data_process(test_file_path, dataset_dirname_prefixs_lst)
		test_iter = DataLoader(
			test_data,
			batch_size=self.batch_size,
			shuffle=self.shuffle,
			collate_fn=self.generate_batch,
		)
		return test_iter

	def generate_batch(self, data_batch):
		coords_batch, features_batch, labels_batch = ME.utils.sparse_collate(
			[d["coordinates"] for d in data_batch],
			[d["features"] for d in data_batch],
			[d["label"] for d in data_batch],
		)
		shapes_batch = torch.stack([d["shape"] for d in data_batch])
		schedule_batch = [d["schedule"] for d in data_batch]
		return coords_batch, features_batch, shapes_batch, schedule_batch, labels_batch


if __name__ == "__main__":
	dataset = LoadHybridSparseMatrixScheduleDataSet()
	data = dataset.load_test_data(os.path.join(root, "dataset", "validation.txt"), ['epyc_7543/spmm'])