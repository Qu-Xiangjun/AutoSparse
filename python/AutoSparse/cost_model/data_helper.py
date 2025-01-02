import os, sys
import time
import logging
import torch


def process_cache():
    """
    The cache modifier for data_process() function.
    """

    def decorating_function(func):
        def wrapper(*args, **kwargs):
            file_path = args[1]
            file_dir = f"{os.sep}".join(file_path.split(os.sep)[:-1])
            file_name = "".join(file_path.split(os.sep)[-1].split(".")[:-1])
            paras = f"cache_{file_name}_"

            schedule_dataset_dir_prefixs = args[2]
            for sdd_prefix in schedule_dataset_dir_prefixs:
                paras += "_".join(sdd_prefix.split(os.sep))

            cache_path = os.path.join(file_dir, paras + ".pt")

            start_time = time.time()
            if not os.path.exists(cache_path):
                logging.info(
                    f"Cache file for data_process {cache_path} don't exist, "
                    "rehandle and cache!"
                )
                data = func(*args, **kwargs)
                with open(cache_path, "wb") as f:
                    torch.save(data, f)
            else:
                logging.info(f"Cache file for data_process {cache_path} loaded!")
                with open(cache_path, "rb") as f:
                    data = torch.load(f)
            end_time = time.time()
            logging.info(f"Data preprocess time is {(end_time - start_time):.3f}s")
            return data

        return wrapper

    return decorating_function
