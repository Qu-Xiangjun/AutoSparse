import os, sys
import random
import torch
import numpy as np
import pandas
from typing import *


class Tokenizer:
    """
    Encode all schedule primitives and sparse matrix feature information into tokens.
    """

    PRIMITIVES = [
        "fsplit",
        "freorder",
        "fmode",
        "lsplit",
        "lreorder",
        "lparallel",
        "lunroll",
        "lvector",
        "openmp_parameter",
        "sparse_info",
    ]

    def __init__(self, embedding_size: int = 128, tensor_name_set: List[str] = None):
        """_summary_

        Parameters
        ----------
        embedding_size : int, optional
            _description_, by default 128
        tensor_name_set : List[str], optional
            All existed tensor name, by default None
        """
        self.embedding_size = embedding_size
        self.primitives_one_hot = {}
        for idx, key in enumerate(self.PRIMITIVES):
            self.primitives_one_hot[key] = [0 for i in range(len(self.PRIMITIVES))]
            self.primitives_one_hot[key][idx] = 1
        if tensor_name_set == None:
            tensor_name_set = [i for i in "ABCDEFGHIJKLMNOPQRST"]
        self.character_mapping = ["<START>", "", " ", "None"] + tensor_name_set
        for c in "ijklmnopqrstuvwxyzabcdefgh":
            self.character_mapping.append(c)
            for id0 in "0123456789":
                self.character_mapping.append(c + id0)
                for id1 in "0123":
                    self.character_mapping.append(c + id0 + id1)

    @staticmethod
    def ScheduleParse(schedule: str) -> List[str]:
        parsed_sch = []
        sch = schedule.split(" ")
        fsplit_count = int(sch[0])
        now_idx = 1
        for idx in range(fsplit_count):
            fsplit_vec = ["fsplit"]
            fsplit_vec.append(sch[now_idx])
            splited_cnt = int(sch[now_idx + 1])
            fsplit_vec.extend(sch[now_idx + 2 : now_idx + 2 + splited_cnt * 2])
            parsed_sch.append(fsplit_vec)  # ["fsplit" i i0 256 i1 16]
            now_idx += 2 + splited_cnt * 2
        sparse_tensor_count = int(sch[now_idx])
        now_idx += 1
        for idx in range(sparse_tensor_count):
            freorder_vec = ["freorder"]
            freorder_vec.append(sch[now_idx])
            axes_size = int(sch[now_idx + 1])
            now_idx += 2
            freorder_vec.extend(sch[now_idx : now_idx + axes_size])
            parsed_sch.append(freorder_vec)  # ["freorder" A i0 i1 k0 k1]
            now_idx += axes_size
        for idx in range(sparse_tensor_count):
            fmode_vec = ["fmode"]
            fmode_vec.append(sch[now_idx])
            axes_size = int(sch[now_idx + 1])
            now_idx += 2
            fmode_vec.extend(sch[now_idx : now_idx + axes_size * 2])
            parsed_sch.append(fmode_vec)  # ["fmode" A i0 1 i1 1 k0 1 k1 1]
            now_idx += axes_size * 2
        lsplit_count = int(sch[now_idx])
        now_idx += 1
        for idx in range(lsplit_count):
            lsplit_vec = ["lsplit"]
            lsplit_vec.append(sch[now_idx])
            splited_cnt = int(sch[now_idx + 1])
            lsplit_vec.extend(sch[now_idx + 2 : now_idx + 2 + splited_cnt * 2])
            parsed_sch.append(lsplit_vec)  # ["lsplit" i i0 256 i1 16]
            now_idx += 2 + splited_cnt * 2
        lreorder_vars_count = int(sch[now_idx])
        now_idx += 1
        # ["lreorder" j0 i0 i1 k0 k1 j1]
        lreorder_vec = ["lreorder"] + sch[now_idx : now_idx + lreorder_vars_count]
        now_idx += lreorder_vars_count
        lparallel_vec = ["lparallel", sch[now_idx]]  # ["lparallel" j0]
        lvector_vec = ["lvector", sch[now_idx + 1]]  # ["lvector" None]
        lunroll_vec = ["lunroll"] + sch[now_idx + 2 : now_idx + 4]  # ["lunroll" k1]
        parsed_sch.extend([lreorder_vec, lparallel_vec, lvector_vec, lunroll_vec])
        now_idx += 4
        openmp_parameter_vec = ["openmp_parameter"] + sch[now_idx : now_idx + 2]
        parsed_sch.append(openmp_parameter_vec)  # ["openmp_parameter" 128 2]

        return parsed_sch

    def BatchPadSequences(
        self, sequences: List[List], max_length: int = None, padding_value=0
    ):
        """Padding of 0 is added to make a batch for the input data sequence.

        Parameters
        ----------
        sequences : List[List]
            Input sequence for every data, which shape is [batch_size, different seq len, embedding size]
        max_length : int, optional
            _description_, by default None
        padding_value : int, optional
            _description_, by default 0

        Returns
        -------
        np.array
            _description_
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                padded_seq = list(seq) + [[padding_value] * self.embedding_size] * (
                    max_length - len(seq)
                )
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)

        return np.array(padded_sequences)

    def EmbedHelper(self, schedule: List[str]) -> List[int]:
        embedded_tokens = []
        for primitive_vec in schedule:
            primitive_name = primitive_vec[0]
            token = self.primitives_one_hot[primitive_name].copy()
            if primitive_name == self.PRIMITIVES[0]:  # ["fsplit" i i0 256 i1 16]
                token.append(self.character_mapping.index(primitive_vec[1]))
                for i in range(2, len(primitive_vec), 2):
                    token.append(self.character_mapping.index(primitive_vec[i]))
                    token.append(int(primitive_vec[i + 1]))
            elif primitive_name == self.PRIMITIVES[1]:  # ["freorder" A i0 i1 k0 k1]
                for i in range(1, len(primitive_vec)):
                    token.append(self.character_mapping.index(primitive_vec[i]))
            elif (
                primitive_name == self.PRIMITIVES[2]
            ):  # ["fmode" A i0 1 i1 1 k0 1 k1 1]
                token.append(self.character_mapping.index(primitive_vec[1]))
                for i in range(2, len(primitive_vec), 2):
                    token.append(self.character_mapping.index(primitive_vec[i]))
                    token.append(int(primitive_vec[i + 1]))
            elif primitive_name == self.PRIMITIVES[3]:  # ["lsplit" i i0 256 i1 16]
                token.append(self.character_mapping.index(primitive_vec[1]))
                for i in range(2, len(primitive_vec), 2):
                    token.append(self.character_mapping.index(primitive_vec[i]))
                    token.append(int(primitive_vec[i + 1]))
            elif primitive_name == self.PRIMITIVES[4]:  # ["lreorder" j0 i0 i1 k0 k1 j1]
                for i in range(1, len(primitive_vec)):
                    token.append(self.character_mapping.index(primitive_vec[i]))
            elif primitive_name == self.PRIMITIVES[5]:  # ["lparallel" j0]
                token.append(self.character_mapping.index(primitive_vec[1]))
            elif primitive_name == self.PRIMITIVES[6]:  # ["lunroll" k1, 4]
                token.append(self.character_mapping.index(primitive_vec[1]))
                token.append(int(primitive_vec[2]))
            elif primitive_name == self.PRIMITIVES[7]:  # ["lvector" None]
                token.append(self.character_mapping.index(primitive_vec[1]))
            elif primitive_name == self.PRIMITIVES[8]:  # ["openmp_parameter" 128 2]
                token.append(int(primitive_vec[1]))
                token.append(int(primitive_vec[2]))
            elif (
                primitive_name == self.PRIMITIVES[9]
            ):  # ["sparse_info" tensor([-0.8154,  0.4300,  0.6851,  0.0416]]
                for item in primitive_vec[1].tolist():
                    assert isinstance(item, float)
                    token.append(item)
            else:
                assert False, f"{primitive_vec[0]} have not in PRIMITIVES"
            assert (
                len(token) <= self.embedding_size
            ), f"embedded token size {len(token)} more than {self.embedding_size}."
            token.extend([0 for i in range(self.embedding_size - len(token))])
            embedded_tokens.append(token)
        return embedded_tokens

    def __call__(
        self, schedules: Union[str, List[str]], sparse_tensor_info: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        schedules : Union[str, List[str]]
            _description_
        sparse_tensor_info : torch.Tensor
            Constructed in [row, col, nonezeros, sparsity]. Notice they usually be normalized.

        Returns
        -------
        torch.Tensor
            Return the batched token sequences.
        """
        # sparse_tensor_info = sparse_tensor_shape: Tuple[int], nonezeros, sparsity
        if isinstance(schedules, str):
            schedules = [schedules]
        elif not isinstance(schedules, List):
            assert False, "Schedules only support type with str and List[str]."

        sequences = []
        for sch in schedules:
            assert isinstance(schedules[0], str), "Schedule must contain str type."
            sch_vecs = self.ScheduleParse(sch)
            sch_vecs.append(["sparse_info", *sparse_tensor_info])
            embedded_tokens = self.EmbedHelper(sch_vecs)
            sequences.append(embedded_tokens)

        return torch.from_numpy(self.BatchPadSequences(sequences)).to(torch.float32)


if __name__ == "__main__":
    schedules = [
        "2 i 2 i0 256 i1 16 k 2 k0 4096 k1 16 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 1 k0 1 k1 1 1 j 2 j0 2 j1 64 6 j0 i0 i1 k0 k1 j1 j0 j1 None 0 128 512 445.855559",
        "2 i 2 i0 32 i1 64 k 2 k0 512 k1 64 1 A 4 k0 i0 k1 i1 A 4 i0 1 i1 0 k0 2 k1 1 1 j 2 j0 8 j1 16 6 j0 j1 k0 i0 k1 i1 j0 i1 j1 16 128 2 309.067448",
        "2 i 2 i0 4096 i1 16 j 2 j0 1024 j1 8 1 A 4 i0 j0 i1 j1 A 4 i0 2 i1 1 j0 4 j1 1 0 4 i0 j0 i1 j1 None None None 0 128 2 5.13304",
    ]
    tokenizer = Tokenizer()
    embedded_tokens = tokenizer(schedules, [0.123, 0.35, 0.13, 0.1284])
    print(embedded_tokens)
