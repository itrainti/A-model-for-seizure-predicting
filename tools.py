import glob
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

FS_IN = 256
FS_OUT = 200
BAND = (0.1, 50.0)
ORDER = 5
MERGE_TOL = 60
WIN_SEC = 10.0
GUARD = 600.0   # 同一切分点两侧丢弃的缓冲秒数
ELECS = ["Fp1-F7", "Fp2-F8", "T7-P7", "T8-P8", "P7-O1", "P8-O2"]

COLS = ["edf_file", "patient_id", "record_id", "channel", "seizure_id", "window_id",
        "start_abs_s", "end_abs_s", "center_abs_s", "raw_start_sample", "raw_end_sample",
        "start_sample", "end_sample", "prev_seizure_id", "next_seizure_id",
        "time_from_previous_seizure_start_s", "time_from_previous_seizure_end_s",
        "time_to_next_seizure_start_s", "time_to_next_seizure_end_s",
        "label", "usable", "excluded_reason"]

_HMS = re.compile(r"File (Start|End) Time:\s*(\d+:\d+:\d+)")
_NSZ = re.compile(r"Number of Seizures in File:\s*(\d+)")
_SZS = re.compile(r"Seizure (?:\d+ )?Start Time:\s*(\d+) seconds")
_SZE = re.compile(r"Seizure (?:\d+ )?End Time:\s*(\d+) seconds")

# 将 HH:MM:SS 转换为秒
def _hms(s):
    h, m, sec = (int(v) for v in s.split(":"))
    return h * 3600 + m * 60 + sec

# 查找指定通道在 EDF 中的位置
def _find(names, ch):
    ch = ch.upper()
    for i, n in enumerate(names):
        if n == ch or n.startswith(ch + "-"):
            return i
    return None

# 获取 EDF 中的全部通道名称
def _channels(path):
    import mne
    return [n.upper() for n in mne.io.read_raw_edf(path, preload=False, verbose="ERROR").ch_names]

# 查找患者目录中的 summary 文件
def _summary(path):
    hits = sorted(glob.glob(os.path.join(path, "*-summary.txt")))
    if not hits:
        raise FileNotFoundError("没找到 summary: %s" % path)
    return hits[0]

# 解析 summary 中的记录和发作时间
def read_summary(path):
    """读取 summary，返回 seizure start/end 等信息"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out, off = [], 0
    for blk in txt.split("File Name:")[1:]:
        name = blk.split("\n")[0].strip()
        ts = _HMS.findall(blk)
        if len(ts) != 2:
            print("[警告] %s: 解析不到 Start/End Time，跳过" % name)
            continue
        t0, t1 = _hms(ts[0][1]), _hms(ts[1][1])
        if t1 <= t0:
            t1 += 86400
        if out and t0 + off < out[-1]["t1"] - 1:
            off += 86400
        t0, t1 = t0 + off, t1 + off
        s = [int(v) for v in _SZS.findall(blk)]
        e = [int(v) for v in _SZE.findall(blk)]
        n = int(_NSZ.search(blk).group(1)) if _NSZ.search(blk) else 0
        if n != len(s) or len(s) != len(e):
            print("[警告] %s: 标注 %d 次发作，解析出 %d 对 start/end" % (name, n, min(len(s), len(e))))
        out.append(dict(file=name, t0=t0, t1=t1,
                        seizures=[(t0 + a, t0 + b) for a, b in zip(s, e)]))
    return out

# 读取指定通道的 EEG 和采样率
def read_edf(path, elec_names: list):
    """读取指定通道的 EEG 和采样率"""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    names = [n.upper() for n in raw.ch_names]
    picks, miss = [], []
    for c in elec_names:
        i = _find(names, c)
        if i is None:
            miss.append(c)
        picks.append(i)
    if miss:
        print("[警告] %s 缺通道 %s，整份 EDF 跳过" % (os.path.basename(path), miss))
        return None, float(raw.info["sfreq"])
    return raw.get_data(picks=picks).astype(np.float64), float(raw.info["sfreq"])

# 对 EEG 进行重采样
def resample_signal(x, rate, target_fs):
    """重采样"""
    from scipy.signal import resample_poly
    return resample_poly(x, int(round(target_fs)), int(round(rate)), axis=-1).astype(np.float32)

# 对 EEG 进行 Butterworth 带通滤波
def bandpass_filter(x, rate, low=0.1, high=50, order=5):
    """Butterworth 带通"""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, np.array([low, high], dtype=np.float64) / (rate / 2.0), btype="bandpass")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)

# 将 EEG 切成固定长度的滑动窗口
def make_windows(x, window_length, window_overlap):
    """切窗口"""
    step = window_length - window_overlap
    if step <= 0 or x.shape[-1] < window_length:
        return np.zeros(x.shape[:-1] + (0, window_length), np.float32), np.zeros(0, int)
    n = (x.shape[-1] - window_length) // step + 1
    starts = np.arange(n) * step
    idx = starts[:, None] + np.arange(window_length)[None, :]
    return np.asarray(x[..., idx], dtype=np.float32), starts

# 根据时间判断窗口标签和对应发作信息
def label_window(start_s, end_s, seizures, SPH, SOP, interictal_thr):
    """
    确定窗口状态及对应发作信息
    {
    "label": 1,
    "seizure_id": 3,
    "prev_seizure_id": 2,
    "next_seizure_id": 3,
    "time_to_next_seizure_start_s": 120.0,
    "time_from_previous_seizure_end_s": 3600.0,
    "usable": 1,
    }
    """
    center = (start_s + end_s) / 2.0
    bounds = []
    for on, off in seizures:
        bounds += [on - SOP - SPH, on - SPH, on, off]

    pid, pon, poff = -1, -1.0, -1.0
    nid, non, noff = -1, -1.0, -1.0
    for i, (on, off) in enumerate(seizures):
        if off <= center:
            pid, pon, poff = i, on, off
        if nid < 0 and on > center:
            nid, non, noff = i, on, off

    d = dict(seizure_id=-1, prev_seizure_id=pid, next_seizure_id=nid,
             time_from_previous_seizure_start_s=(center - pon) if pid >= 0 else -1.0,
             time_from_previous_seizure_end_s=(center - poff) if pid >= 0 else -1.0,
             time_to_next_seizure_start_s=(non - center) if nid >= 0 else -1.0,
             time_to_next_seizure_end_s=(noff - center) if nid >= 0 else -1.0)

    if any(start_s < b < end_s for b in bounds):
        d.update(label=3, usable=0, excluded_reason="overlap")
        return d

    for i, (on, off) in enumerate(seizures):
        if on <= center < off:
            d.update(label=2, usable=1, excluded_reason="none", seizure_id=i)
            return d
        if on - SOP - SPH <= center < on - SPH:
            d.update(label=1, usable=1, excluded_reason="none", seizure_id=i)
            return d
        if on - SPH <= center < on:
            d.update(label=3, usable=0, excluded_reason="sph", seizure_id=i)
            return d

    far_next = nid < 0 or d["time_to_next_seizure_start_s"] > interictal_thr
    far_prev = pid < 0 or d["time_from_previous_seizure_end_s"] > interictal_thr
    if far_next and far_prev:
        d.update(label=0, usable=1, excluded_reason="none")
    else:
        d.update(label=3, usable=0, excluded_reason="exclusion")
    return d

# 合并时间上连续的 EDF 记录
def sessions(recs, tol=MERGE_TOL):
    out = []
    for r in recs:
        if out and r["t0"] - out[-1]["t1"] <= tol:
            out[-1].update(t1=r["t1"], files=out[-1]["files"] + [r["file"]])
        else:
            out.append(dict(t0=r["t0"], t1=r["t1"], files=[r["file"]]))
    return out

# 筛选满足实验条件的有效发作
def valid_seizures(recs, SPH, SOP, interictal_thr):
    seize = sorted(s for r in recs for s in r["seizures"])
    sess = sessions(recs)
    out = []
    for i, (on, off) in enumerate(seize):
        s = next((x for x in sess if x["t0"] <= on < x["t1"]), None)
        if s and on - (SPH + SOP) >= s["t0"] and on + interictal_thr <= s["t1"]:
            out.append(i)
    return out

# 检查所有目标通道是否在患者的 EDF 中存在
def channel_availability(path, elec_names):
    cache, out = {}, {}
    for c in elec_names:
        out[c] = True
    for r in read_summary(_summary(path)):
        if r["file"] not in cache:
            cache[r["file"]] = _channels(os.path.join(path, r["file"]))
        for c in elec_names:
            if _find(cache[r["file"]], c) is None:
                out[c] = False
    return out

# 打印窗口和标签统计结果
def _print_counts(m, tag):
    print("%s 原始窗 %d" % (tag, len(m)))
    for lb in (0, 1, 2, 3):
        print("%s   label=%d %d" % (tag, lb, int((m.label == lb).sum())))
    for r in ("overlap", "sph", "exclusion"):
        print("%s   excluded(%s) %d" % (tag, r, int((m.excluded_reason == r).sum())))
    print("%s usable %d / excluded %d" % (tag, int(m.usable.sum()), int((m.usable == 0).sum())))

# 完成 EEG 预处理、切窗和标签生成
def preprocess(path, SPH, SOP, interictal_thr, window_length, window_overlap, elec_names):
    """
    交界处的窗口默认不保留
    :param path: 包括edf和summary的文件夹路径
    :param SPH: 发作前留给患者的准备时间
    :param SOP: 发作前期
    :param interictal_thr: 发作间期里发作期的最低时间
    :param window_length: 窗口长度
    :param window_overlap: 窗口重叠
    :param window_step: 滑动步长
    :param elec_names: 需要用到的电极
    :return:windows、metadata
    """
    pid = os.path.basename(os.path.normpath(path))
    recs = read_summary(_summary(path))
    seize = sorted(s for r in recs for s in r["seizures"]) # 从dataframe取出发作[s,e]

    W = int(round(window_length * FS_OUT))
    O = int(round(window_overlap * FS_OUT))
    rows = {c: [] for c in elec_names}
    wins = {c: [] for c in elec_names}
    # skipped：存储该跳过的文件
    # short：存储长度不够的窗口
    skipped, short = [], []

    # 读取各edf文件数据
    for rec in recs:
        x, rate = read_edf(os.path.join(path, rec["file"]), elec_names)
        if x is None: #若为空则跳过
            skipped.append(rec["file"])
            continue
        # 滤波
        x = bandpass_filter(resample_signal(x, rate, FS_OUT), FS_OUT, *BAND, ORDER)
        # 切窗
        win, starts = make_windows(x, W, O)
        # 跳过没有切窗成功的文件
        if not len(starts):
            short.append(rec["file"])
            continue

        rest = x.shape[-1] - (int(starts[-1]) + W) # 计算切窗剩余长度
        if rest > 0:
            short.append("%s(丢 %.1fs)" % (rec["file"], rest / FS_OUT))

        # 打标签
        labs = [label_window(rec["t0"] + s / FS_OUT, rec["t0"] + (s + W) / FS_OUT,
                             seize, SPH, SOP, interictal_thr) for s in starts]
        # 分别写出各通道metadata
        for i, c in enumerate(elec_names):
            n0 = len(rows[c])
            for k, s in enumerate(starts):
                d = labs[k]
                rows[c].append(dict(
                    edf_file=rec["file"], patient_id=pid, record_id=rec["file"].replace(".edf", ""),
                    channel=c, seizure_id=int(d["seizure_id"]), window_id=n0 + k,
                    start_abs_s=round(rec["t0"] + s / FS_OUT, 3),
                    end_abs_s=round(rec["t0"] + (s + W) / FS_OUT, 3),
                    center_abs_s=round(rec["t0"] + (s + W / 2.0) / FS_OUT, 3),
                    raw_start_sample=int(round(s * FS_IN / FS_OUT)),
                    raw_end_sample=int(round((s + W) * FS_IN / FS_OUT)),
                    start_sample=int(s), end_sample=int(s + W),
                    prev_seizure_id=int(d["prev_seizure_id"]), next_seizure_id=int(d["next_seizure_id"]),
                    time_from_previous_seizure_start_s=round(d["time_from_previous_seizure_start_s"], 3),
                    time_from_previous_seizure_end_s=round(d["time_from_previous_seizure_end_s"], 3),
                    time_to_next_seizure_start_s=round(d["time_to_next_seizure_start_s"], 3),
                    time_to_next_seizure_end_s=round(d["time_to_next_seizure_end_s"], 3),
                    label=int(d["label"]), usable=int(d["usable"]),
                    excluded_reason=d["excluded_reason"]))
            wins[c].append(win[i])

    if skipped:
        print("[警告] %s 跳过 %d 份 EDF: %s" % (pid, len(skipped), skipped))
    if short:
        print("[边界] %s 末尾不足整窗: %s" % (pid, short))

    windows, meta = {}, {}
    for c in elec_names:
        # 把一个通道的所有窗口按dim=0（窗口）拼接成一个完整窗口数组
        windows[c] = np.concatenate(wins[c]) if wins[c] else np.zeros((0, W), np.float32)
        # 把所有元数据拼接成一个metadata
        meta[c] = pd.DataFrame(rows[c], columns=COLS)

    # 打印结果
    _print_counts(meta[elec_names[0]], "[%s]" % pid)
    for c in elec_names:
        print("[%s] %s 窗 %d" % (pid, c, len(meta[c])))

    return windows, meta

# 统计患者的发作和有效窗口数量
def patient_statistics(root, elec_names, SPH=120, SOP=1800, interictal_thr=120 * 60,
                       window_length=10, window_overlap=5, counts=True, post_thr=180 * 60,
                       out="patient_statistics.csv"):
    rows = []
    for pid in sorted(os.listdir(root)):
        path = os.path.join(root, pid)
        if not pid.startswith("chb") or not os.path.isdir(path):
            continue
        try:
            recs = read_summary(_summary(path))
        except FileNotFoundError as e:
            print("[警告] %s" % e)
            continue
        valid = valid_seizures(recs, SPH, SOP, post_thr)
        av = channel_availability(path, elec_names)
        row = dict(patient_id=pid, n_edf=len(recs),
                   n_seizure=sum(len(r["seizures"]) for r in recs), n_valid_seizure=len(valid))
        for c in elec_names:
            row["ch_" + c] = int(av[c])
        if counts and valid:
            _, meta = preprocess(path, SPH, SOP, interictal_thr, window_length, window_overlap, elec_names)
            m = meta[elec_names[0]]
            row.update(n_windows=len(m),
                       n_interictal=int((m.label == 0).sum()),
                       n_preictal=int((m.label == 1).sum()),
                       n_ictal=int((m.label == 2).sum()),
                       n_overlap=int((m.excluded_reason == "overlap").sum()),
                       n_excluded=int((m.label == 3).sum()),
                       n_usable=int(m.usable.sum()))
        rows.append(row)
        print("[统计] %s EDF %d 发作 %d 有效 %d" % (pid, row["n_edf"], row["n_seizure"], row["n_valid_seizure"]))

    df = pd.DataFrame(rows).fillna(0).sort_values(
        ["n_valid_seizure", "n_seizure"], ascending=False).reset_index(drop=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print("[统计] 写出 %s，共 %d 个病人" % (out, len(df)))
    return df

def _owner(nid, seizes):
    return np.where(nid < 0, seizes[-1], nid)


def _pick(pool, k, seed):
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= len(pool):
        return np.asarray(pool)
    return np.random.default_rng(seed).choice(pool, k, replace=False)


def _split_set(pre_idx, int_pool, tag, seed, balance=True):
    if not balance:
        idx = np.concatenate([pre_idx, int_pool]) if len(pre_idx) + len(int_pool) else np.zeros(0, np.int64)
        return idx, len(pre_idx), len(int_pool)
    k = min(len(pre_idx), len(int_pool))
    if k < len(pre_idx):
        print("[警告] %s: interictal 只够 %d，preictal %d，无法平衡" % (tag, k, len(pre_idx)))
    idx = np.concatenate([pre_idx, _pick(int_pool, k, seed)]) if len(pre_idx) else np.zeros(0, np.int64)
    return idx, len(pre_idx), k


def _fold(name, tr, va, te, seed, balance=True):
    a, ap, ai = _split_set(tr[0], tr[1], "%s train" % name, seed, balance)
    b, bp, bi = _split_set(va[0], va[1], "%s val" % name, seed + 1, balance)
    c, cp, ci = _split_set(te[0], te[1], "%s test" % name, seed + 2, balance)
    return dict(seizure_id=name, train_idx=a, val_idx=b, test_idx=c,
                n_train_pre=ap, n_train_inter=ai,
                n_val_pre=bp, n_val_inter=bi,
                n_test_pre=cp, n_test_inter=ci)


def _split_growing(metadata, val, mode, base, seed):
    index = metadata.index.to_numpy()
    label = metadata.label.to_numpy()
    sid = metadata.seizure_id.to_numpy()
    cen = metadata.center_abs_s.to_numpy()
    ttn = metadata.time_to_next_seizure_start_s.to_numpy()

    pre = (label == 1)
    seizes = sorted(set(sid[pre].tolist()))
    onset = {}
    for s in seizes:
        g = cen[pre & (sid == s)] + ttn[pre & (sid == s)]
        onset[s] = float(g.max())

    folds = []
    for n in seizes:
        cut = onset[n]
        tr_sz = [s for s in seizes if s < n]
        if mode == "fixed":
            tr_sz = tr_sz[:base]
        if not tr_sz:
            continue

        te_pre = index[pre & (sid == n)]
        tr_pre = index[pre & np.isin(sid, tr_sz)]

        pool = index[(label == 0) & (cen < cut)]
        pool = pool[np.argsort(cen[pool])]

        k_te = min(len(te_pre), len(pool))
        if k_te < len(te_pre):
            print("[警告] 发作 %d: interictal 池 %d < 测试 preictal %d" % (n, len(pool), len(te_pre)))
        te_int = pool[len(pool) - k_te:] if k_te else pool[:0]
        head = pool[:len(pool) - k_te]
        if k_te:
            head = head[cen[head] <= cen[te_int].min() - 2 * WIN_SEC]

        k_tr = min(len(tr_pre), len(head))
        if k_tr < len(tr_pre):
            print("[警告] 发作 %d: 训练 interictal 只够 %d，preictal %d" % (n, k_tr, len(tr_pre)))
        rng = np.random.default_rng(seed + n)
        tr_int = rng.choice(head, k_tr, replace=False) if k_tr else head[:0]

        va_pre, va_int = tr_pre[:0], tr_int[:0]
        if val:
            k = int(min(len(tr_pre), len(tr_int)) * val)
            if k:
                va_pre = tr_pre[np.argsort(cen[tr_pre])][len(tr_pre) - k:]
                va_int = tr_int[np.argsort(cen[tr_int])][len(tr_int) - k:]
                tr_pre = np.setdiff1d(tr_pre, va_pre)
                tr_int = np.setdiff1d(tr_int, va_int)

        folds.append(dict(seizure_id=n, train_idx=np.concatenate([tr_pre, tr_int]),
                          val_idx=np.concatenate([va_pre, va_int]),
                          test_idx=np.concatenate([te_pre, te_int]),
                          n_train_pre=len(tr_pre), n_train_inter=len(tr_int),
                          n_val_pre=len(va_pre), n_val_inter=len(va_int),
                          n_test_pre=len(te_pre), n_test_inter=len(te_int)))
    return folds


def _split_loocv(metadata, val, seed, enc_val=None):
    index = metadata.index.to_numpy()
    label = metadata.label.to_numpy()
    sid = metadata.seizure_id.to_numpy()
    prev = metadata.prev_seizure_id.to_numpy()
    nxt = metadata.next_seizure_id.to_numpy()
    cen = metadata.center_abs_s.to_numpy()

    pre = (label == 1)
    bg = (label == 0)
    seizes = sorted(set(sid[pre].tolist()))
    onset = {k: float(cen[pre & (sid == k)].max()) for k in seizes}

    # 间期池按 (前一次发作, 后一次发作) 分块：块内连续，块间至少隔一次发作 + 两侧 interictal_thr
    groups = {}
    for i, key in zip(index[bg], zip(prev[bg], nxt[bg])):
        groups.setdefault(key, []).append(i)
    blocks = sorted(groups.values(), key=lambda v: cen[v].min())
    blocks = [np.asarray(v, dtype=np.int64) for v in blocks]
    # 块不够三方分时，从最长的块中间切开，砍掉 GUARD 秒缓冲，保证三边都拿得到间期
    while len(blocks) < 3:
        i = max(range(len(blocks)), key=lambda j: len(blocks[j]))
        mid = cen[blocks[i][len(blocks[i]) // 2]]
        left = blocks[i][cen[blocks[i]] <= mid - GUARD]
        right = blocks[i][cen[blocks[i]] >= mid + GUARD]
        if not len(left) or not len(right):
            break
        blocks[i:i + 1] = [left, right]
    blocks = sorted(blocks, key=lambda v: cen[v].min())

    folds = []
    for k in seizes:
        rest = [s for s in seizes if s != k]
        n_val = max(1, int(round(val * len(rest)))) if val and rest else 0
        near = sorted(rest, key=lambda s: (abs(s - k), s > k))
        va_sz, tr_sz = sorted(near[:n_val]), sorted(near[n_val:])

        # 间期块按离测试发作的远近排序，近的给测试、其次验证，其余训练；整块分配，一块不跨集合
        order = sorted(range(len(blocks)),
                       key=lambda b: (abs(float(cen[blocks[b]].mean()) - onset[k]), b))
        need = [len(index[pre & (sid == k)]),
                len(index[pre & np.isin(sid, va_sz)]) if va_sz else 0]
        picked = [[], [], []]
        b = 0
        for j in range(2):  # 测试、验证依次取，剩下的全给训练
            while len(picked[j]) < need[j] and b < len(blocks) - (2 - j):
                picked[j] += blocks[order[b]].tolist()
                b += 1
        for bb in order[b:]:
            picked[2] += blocks[bb].tolist()

        te = (index[pre & (sid == k)], np.asarray(picked[0], dtype=np.int64))
        va = (index[pre & np.isin(sid, va_sz)], np.asarray(picked[1], dtype=np.int64)) \
            if va_sz else (index[:0], index[:0])
        tr = (index[pre & np.isin(sid, tr_sz)], np.asarray(picked[2], dtype=np.int64))

        f = _fold(k, tr, va, te, seed + k)
        pool = np.setdiff1d(index, np.concatenate([te[0], te[1]]))
        pool = pool[np.argsort(cen[pool])]
        m = int(round(len(pool) * enc_val)) if enc_val else 0
        f["auto_encoder_idx"] = pool[:len(pool) - m]
        f["auto_encoder_val_idx"] = pool[len(pool) - m:]
        folds.append(f)
    return folds


def _split_temporal(metadata, val, seed, test_ratio=0.35):
    index = metadata.index.to_numpy()
    label = metadata.label.to_numpy()
    sid = metadata.seizure_id.to_numpy()

    pre = (label == 1)
    bg = (label == 0)
    seizes = sorted(set(sid[pre].tolist()))
    own = _owner(metadata.next_seizure_id.to_numpy(), seizes)
    n_test = max(1, int(round(test_ratio * len(seizes))))
    tr_sz, te_sz = seizes[:len(seizes) - n_test], seizes[len(seizes) - n_test:]
    n_val = max(1, int(round(val * len(tr_sz)))) if val and tr_sz else 0
    va_sz, tr_sz = tr_sz[len(tr_sz) - n_val:], tr_sz[:len(tr_sz) - n_val]

    print("[划分] 共 %d 次发作，最后 %d 次用作测试 %s，验证 %s"
          % (len(seizes), n_test, te_sz, va_sz or "无"))

    te = (index[pre & np.isin(sid, te_sz)], index[bg & np.isin(own, te_sz)])
    va = (index[pre & np.isin(sid, va_sz)], index[bg & np.isin(own, va_sz)])
    tr = (index[pre & np.isin(sid, tr_sz)], index[bg & np.isin(own, tr_sz)])
    return [_fold(-1, tr, va, te, seed, balance=False)]


# 按发作时间顺序划分训练集、验证集和测试集
# method=LOOCV 按发作聚类窗口，train/val/test 各自只由完整的发作组成
# method=temporal 按 FGL 口径，最后 35% 次发作整体作为测试
# method=growingCV 为原来的因果扩张口径
def split_train_test(metadata, val: float = None, mode="expanding", base=1, seed=0,
                     method="growingCV"):
    """

    :param val: 是否划分验证集，以及所占比例。float 表示下游模型的验证集比例；
                二元组 (enc_val, model_val) 时第一个是自编码器的验证集比例，第二个是下游模型的
    :param method: growingCV / LOOCV / temporal。LOOCV 额外返回 auto_encoder_idx 和
                   auto_encoder_val_idx（该折测试发作以外的全部窗，供每折重训自编码器）
    :return:
    """
    if method not in ("growingCV", "LOOCV", "temporal"):
        raise ValueError("method 只能是 growingCV / LOOCV / temporal")
    if mode not in ("expanding", "fixed"):
        raise ValueError("mode 只能是 expanding 或 fixed")

    enc_val = None
    if isinstance(val, (tuple, list)):
        val = list(val)
        enc_val = val[0]
        val = val[1] if len(val) > 1 else None

    if method == "LOOCV":
        folds = _split_loocv(metadata, val, seed, enc_val)
    elif method == "temporal":
        folds = _split_temporal(metadata, val, seed)
    else:
        folds = _split_growing(metadata, val, mode, base, seed)

    print("[划分] method=%s%s 共 %d 折"
          % (method, " mode=%s" % mode if method == "growingCV" else "", len(folds)))
    fmt = "[划分] %-8s %-14s %-14s %-14s"
    head = ("seizure", "train(pre/int)", "val(pre/int)", "test(pre/int)")
    if method == "LOOCV":
        fmt += " %-16s"
        head += ("autoenc(train/val)",)
    print(fmt % head)
    for f in folds:
        row = ("S%d" % f["seizure_id"] if f["seizure_id"] >= 0 else "all",
               "%d/%d" % (f["n_train_pre"], f["n_train_inter"]),
               "%d/%d" % (f["n_val_pre"], f["n_val_inter"]),
               "%d/%d" % (f["n_test_pre"], f["n_test_inter"]))
        if method == "LOOCV":
            row += ("%d/%d" % (len(f["auto_encoder_idx"]), len(f["auto_encoder_val_idx"])),)
        print(fmt % row)
    return folds


def train_val_test(elec_names, meta, windows, seed, val=(0.1, 0.2), method="LOOCV"):
    """按电极、按折切分好的张量，dataset[elec][fold][key]"""
    dataset = []
    for elec in elec_names:
        data_set = split_train_test(meta[elec], seed=seed, val=val, method=method)
        elec_list = []
        for fold in range(len(data_set)):
            encoder_train_idx = data_set[fold]["auto_encoder_idx"]
            encoder_val_idx = data_set[fold]["auto_encoder_val_idx"]
            train_idx = data_set[fold]["train_idx"]
            val_idx = data_set[fold]["val_idx"]
            test_idx = data_set[fold]["test_idx"]
            wins_and_label = {}

            # 原始窗口 -> 升维 -> 张量 float32
            wins_and_label["encoder_train_wins"] = torch.tensor(windows[elec][encoder_train_idx][:, None, :], dtype=torch.float32)
            wins_and_label["encoder_val_wins"] = torch.tensor(windows[elec][encoder_val_idx][:, None, :], dtype=torch.float32)
            wins_and_label["train_wins"] = torch.tensor(windows[elec][train_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_train"] = torch.tensor(meta[elec].iloc[train_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)
            wins_and_label["val_wins"] = torch.tensor(windows[elec][val_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_val"] = torch.tensor(meta[elec].iloc[val_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)
            wins_and_label["test_wins"] = torch.tensor(windows[elec][test_idx][:, None, :], dtype=torch.float32)
            wins_and_label["y_test"] = torch.tensor(meta[elec].iloc[test_idx].label.to_numpy(), dtype=torch.float32).unsqueeze(1)

            elec_list.append(wins_and_label)

        dataset.append(elec_list)
    return dataset


def train_elec_model(model, optimizer, loss_function, dataset, elec_idx, elec_names, weight_dir,
                     device="cuda", patience=5, epochs=30, batch_size=32):
    """训练单电极模型，返回 (权重路径列表, 日志字典)"""
    model.to(device)
    now = datetime.now().strftime("%y%m_%d_%H%M%S")
    weight_dir = Path(weight_dir)
    fold_paths = []
    fold_logs = {}

    for fold in range(len(dataset[elec_idx])):
        train_dataloader = DataLoader(TensorDataset(dataset[elec_idx][fold]["train_wins"], dataset[elec_idx][fold]["y_train"]), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(dataset[elec_idx][fold]["val_wins"], dataset[elec_idx][fold]["y_val"]), batch_size=batch_size, shuffle=False)

        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None
        fold_logs[f"fold_{fold + 1}"] = {"epochs": []}

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            val_loss = 0

            model.train()
            for (x, y) in train_dataloader:
                x, y = x.to("cuda"), y.to("cuda")
                optimizer.zero_grad()
                out = model(x)
                loss = loss_function(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)

            model.eval()
            with torch.no_grad():
                val_targets = []
                val_probs = []
                for (x, y) in val_dataloader:
                    x, y = x.to("cuda"), y.to("cuda")
                    out = model(x)
                    loss = loss_function(out, y)
                    val_loss += loss.item()
                    probs = torch.sigmoid(out)
                    val_probs.extend(probs.cpu().numpy().flatten())
                    val_targets.extend(y.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_dataloader)
            val_auc = roc_auc_score(val_targets, val_probs)

            end_time = time.time()
            fold_logs[f"fold_{fold + 1}"]["epochs"].append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "val_auc": round(val_auc, 4),
                "time": round(end_time - start_time, 2)
            })

            print(f"Epoch [{epoch+1:02d}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.2e} | "
                  f"Val Loss: {avg_val_loss:.2e} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Time: {end_time - start_time:.1f}s")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Early Stopping")
                    fold_logs[f"fold_{fold + 1}"]["early_stop_epoch"] = epoch + 1
                    break

        if best_model_state is not None:
            path = weight_dir / f"{now}_{elec_names[elec_idx]}_fold{fold+1}.pth"
            fold_paths.append(path)
            torch.save(best_model_state, path)
            fold_logs[f"fold_{fold + 1}"]["weight_path"] = path
            fold_logs[f"fold_{fold + 1}"]["best_val_loss"] = round(best_val_loss, 6)

    return fold_paths, fold_logs


def train_elec_atten(dataset, path, **kwargs):
    """用各电极的验证集概率训练电极级融合模型，返回每折一个 ElecAfterMultiHeadAttention"""
    from models import ConvBILSTMClassify, ElecAfterMultiHeadAttention

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        weights = data["weights"]
        config = data["config"]

    n_folds = len(weights) // 6
    weights_folds = [[] for _ in range(n_folds)]
    for p in weights:
        p = Path(p)
        for i in range(n_folds):
            if f"fold{i+1}" in p.name:
                weights_folds[i].append(p)

    fusion_models = []
    for i, wf_list in enumerate(weights_folds):
        print(f"正在训练第{i}折的融合模型...")

        models = []
        for j, wf in enumerate(wf_list):
            model = ConvBILSTMClassify(
                channal_attention=config.get("channal_attention", False),
                temporal_attention=config.get("temporal_attention", False),
                dropout=config.get("dropout", 0.5),
                hidden_size=config.get("hidden_size", 10)
            ).to("cuda")
            model.load_state_dict(torch.load(wf))
            model.eval()
            models.append(model)

        val_elec_probs = []
        val_targets = None
        for j in range(6):
            val_loader = DataLoader(TensorDataset(dataset[j][i]["val_wins"], dataset[j][i]["y_val"]), batch_size=64, shuffle=False)
            probs, targets = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    p = models[j](x.to("cuda")).sigmoid().cpu().numpy().flatten()
                    probs.extend(p)
                    targets.extend(y.numpy().flatten())
            val_elec_probs.append(probs)
            val_targets = targets

        val_tensor = torch.tensor(np.array(val_elec_probs).T, dtype=torch.float32).unsqueeze(-1).cuda()
        y_val_tensor = torch.tensor(val_targets, dtype=torch.float32).cuda().unsqueeze(-1)

        elec_atten = ElecAfterMultiHeadAttention().to("cuda")
        optimizer = torch.optim.Adam(elec_atten.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        for epoch in range(50):
            elec_atten.train()
            optimizer.zero_grad()
            preds = elec_atten(val_tensor)
            loss = criterion(preds, y_val_tensor)
            loss.backward()
            optimizer.step()

        elec_atten.eval()
        fusion_models.append(elec_atten)

    return fusion_models


def evaluate_classification(y_true, y_prob, threshold=0.5):
    """
    计算二分类模型的各种评估指标
    :param y_true: 真实标签 (numpy array 或 list, 形状 1D)
    :param y_prob: 预测概率 (numpy array 或 list, 形状 1D)
    :param threshold: 判定阳性的阈值，默认 0.5
    :return: 包含各项指标的字典
    """
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)

    # 敏感度 (Sensitivity) = 召回率 (Recall) = TP / (TP + FN)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    # 特异度 (Specificity) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sensitivity + specificity) / 2  # 平衡准确率

    print("=" * 40)
    print(f"📊 分类评估指标 (Threshold = {threshold})")
    print("=" * 40)
    print(f"测试集样本数: {len(y_true)} (阳性 {tp+fn} / 阴性 {tn+fp})")
    print(f"AUC-ROC     : {auc:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Sensitivity : {sensitivity:.4f} (Recall)")
    print(f"Specificity : {specificity:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print("-" * 40)
    print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 40)

    # json.dump：只能写入python的内建类型
    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "balanced_acc": float(bal_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def parse_date(date):
    """
    date:如 261028
    """
    date = str(date)
    if len(date) != 6:
        print("请输入正确格式的日期，如'261028'")
        return None
    year_month = date[:4]
    day = date[-2:]
    ym_d = year_month + "_" + day
    return ym_d


def match_config(config: dict, query: dict):
    """config：日志中的配置；query：查询配置"""
    for k, v in query.items():
        if k not in config:
            print(f"config中没有参数:{k}")
            return False
        config_v = config[k]
        # 只有数字才比较；bool是一种特殊的int，因此要额外排除
        if isinstance(config_v, (int, float)) and not isinstance(config_v, bool) \
           and isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isclose(config_v, v, rel_tol=1e-4):
                continue
            else:
                return False
        else:
            if config_v != v:
                return False
    return True


def find_weights(log_dir, date=None, **kwargs):
    """在 log_dir 下按配置筛权重清单，返回 [{"path","config","n_weights","mtime"}]，按 mtime 降序"""
    log_dir = Path(log_dir)
    if date:
        json_paths = [
            p for p in log_dir.glob("*.json")
            if parse_date(date) and parse_date(date) in p.name
        ]
    else:
        json_paths = [p for p in log_dir.glob("*.json")]

    matches = []
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            print("json解析失败")
            continue

        if not match_config(data["config"], kwargs):
            continue

        matches.append({
            "path": p,
            "config": data["config"],
            "n_weights": len(data.get("weights", [])),
            "mtime": p.stat().st_mtime,
        })

    matches.sort(key=lambda x: x["mtime"], reverse=True)
    return matches


def test_elec_model(dataset, fusion_models, path=None, date=None, log_dir=None, **kwargs):
    """按 manifest 里的权重测每一折，单电极和电极融合的指标都写回 manifest 的 test_index"""
    from models import ConvBILSTMClassify

    if not path:
        matches = find_weights(log_dir, date=date, **kwargs)
        if len(matches) != 1:
            print("请增加限制参数，matches中只能有一个路径")
            return None
        path = matches[0]["path"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            weights = data["weights"]
            config = data["config"]
    except:
        print("无法打开json文件")
        return None

    if "test_index" not in data:
        data["test_index"] = []

    n_folds = len(weights) // 6
    weights_folds = [[] for _ in range(n_folds)]
    for p in weights:
        p = Path(p)
        for i in range(n_folds):
            if f"fold{i+1}" in p.name:
                weights_folds[i].append(p)

    for i, wf_list in enumerate(weights_folds):
        print(f"正在测试第{i}折")
        results = []
        results_atten = []

        all_elec_probs = []
        all_targets = None

        for j, wf in enumerate(wf_list):
            conv_bi_lstm = ConvBILSTMClassify(
                channal_attention=config.get("channal_attention", False),
                temporal_attention=config.get("temporal_attention", False),
                dropout=config.get("dropout", 0.5),
                hidden_size=config.get("hidden_size", 10)
            ).to("cuda")
            conv_bi_lstm.load_state_dict(torch.load(wf))

            test_dataloader = DataLoader(TensorDataset(
                dataset[j][i]["test_wins"],
                dataset[j][i]["y_test"]
            ), batch_size=32, shuffle=False)

            all_probs = []
            targets = []
            conv_bi_lstm.eval()
            with torch.no_grad():
                for (x, y) in test_dataloader:
                    x, y = x.to("cuda"), y.to("cuda")
                    probs = conv_bi_lstm(x).sigmoid()
                    all_probs.extend(probs.cpu().numpy().flatten())
                    targets.extend(y.cpu().numpy().flatten())

            metrics = evaluate_classification(targets, all_probs)
            results.append(metrics)

            all_elec_probs.append(all_probs)
            all_targets = targets

        all_probs_tensor = torch.tensor(np.array(all_elec_probs).T, dtype=torch.float32).unsqueeze(-1).cuda()
        elec_atten = fusion_models[i]
        with torch.no_grad():
            all_probs_atten = elec_atten(all_probs_tensor).cpu().numpy().flatten()

        metrics_atten = evaluate_classification(all_targets, all_probs_atten)
        results_atten.append(metrics_atten)

        data["test_index"].append({
            "fold": i,
            "metrics": results,
            "metrics_atten": results_atten
        })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
