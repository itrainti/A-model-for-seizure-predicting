import os

import h5py
import numpy as np
import pandas as pd

# 把元数据 DataFrame 保存为 CSV
def save_metadata(path, df):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

# 把嵌入向量和行号存成 HDF5 文件
def save_embeddings(path, emb, row_id, attrs=None):
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    row_id = np.ascontiguousarray(row_id, dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("embeddings", data=emb)
        f.create_dataset("row_id", data=row_id)
        f.attrs["embedding_dim"] = int(emb.shape[1]) if emb.ndim == 2 else 0
        f.attrs["n_windows"] = int(emb.shape[0])
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    os.replace(tmp, path)

# 从 HDF5 文件读回嵌入向量和行号
def load_embeddings(path):
    with h5py.File(path, "r") as f:
        return f["embeddings"][:], f["row_id"][:]

# 读取 HDF5 文件里记录的嵌入维度
def embedding_dim(path):
    with h5py.File(path, "r") as f:
        return int(f.attrs["embedding_dim"])

# 检查嵌入文件是否存在，并可指定维度是否匹配
def exists(path, dim=None):
    if not os.path.exists(path):
        return False
    if dim is None:
        return True
    try:
        return embedding_dim(path) == int(dim)
    except Exception:
        return False

# 从 CSV 读回元数据 DataFrame
def load_metadata(path):
    return pd.read_csv(path)
