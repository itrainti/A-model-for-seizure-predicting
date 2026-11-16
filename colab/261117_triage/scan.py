"""在 meta_cache 上扫参数：窗长 / 重叠 / 间期阈值。

只读 meta_cache（每个 EDF 的 n_times、sfreq、ch_names + summary 的发作时刻），
不碰任何 EDF。输出「参数 × 患者」的窗数、折数、最终能否训练。

用法：
    python scan.py                 # 全量扫描
    python scan.py --pilot         # 只跑 3 个患者 + 少量组合，测时间
    python scan.py --validate      # 和 results/*.json 的真实结果对拍
"""
import argparse
import contextlib
import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
META = HERE / "meta_cache"
RESULTS = HERE / "results"

FS_OUT = 200
SPH, SOP = 2 * 60, 30 * 60
ENC_VAL, CLS_VAL = 0.1, 0.1
SEED = 0
MIN_FOLDS = 3
ELECS = ["Fp1-F7", "Fp2-F8", "T7-P7", "T8-P8", "P7-O1", "P8-O2"]

WIN_LENS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]          # 秒
OVERLAPS = [0.0, 0.25, 0.5, 0.75]                               # 占窗长的比例
INTER_THRS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]                # 小时


def load_tools():
    spec = importlib.util.spec_from_file_location("t", str(HERE / "tools.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t = load_tools()


def _find(names, ch):
    return t._find(names, ch)


# ---------------------------------------------------------------- 窗口与标签
def build_windows(cache, win_len_s, overlap, elecs, sph=None, sop=None):
    """返回每个窗的 (start_abs, center_abs, prev_id, next_id, base_label, seizure_id)

    base_label: 1/2/3 与 INTER_THR 无关；4 表示「远/近发作，待定」，由 INTER_THR 决定 0 或 3
    完全复刻 t.label_window 的判定顺序（边界重叠 → 逐发作 ictal/preictal/sph → 其余）
    sph/sop 不给就用模块默认（秒）
    """
    sph = SPH if sph is None else sph
    sop = SOP if sop is None else sop
    W = int(round(win_len_s * FS_OUT))
    O = int(round(win_len_s * overlap * FS_OUT))
    step = W - O
    if step <= 0:
        return None

    seize = sorted(s for rec in cache["summary"] for s in rec["seizures"])
    if not seize:
        return None
    onsets = np.array([s[0] for s in seize], dtype=np.float64)
    offsets = np.array([s[1] for s in seize], dtype=np.float64)
    bounds = np.sort(np.concatenate([onsets - sop - sph, onsets - sph, onsets, offsets]))

    parts = []
    for rec in cache["summary"]:
        info = cache["edf"].get(rec["file"])
        if info is None or info.get("error"):
            continue
        names = [n.upper() for n in info["ch_names"]]
        if any(_find(names, c) is None for c in elecs):
            continue
        L = -(-int(info["n_times"]) * FS_OUT // int(round(info["sfreq"])))
        if L < W:
            continue
        n = (L - W) // step + 1
        starts_local = np.arange(n, dtype=np.int64) * step
        start_abs = rec["t0"] + starts_local / FS_OUT
        end_abs = rec["t0"] + (starts_local + W) / FS_OUT
        center = (start_abs + end_abs) / 2.0

        pid = np.searchsorted(offsets, center, side="right") - 1
        nid = np.searchsorted(onsets, center, side="right")
        nid_ok = nid < len(onsets)
        prev_out = np.where(pid >= 0, pid, -1).astype(np.int32)
        next_out = np.where(nid_ok, nid, -1).astype(np.int32)

        labels = np.zeros(n, dtype=np.int8)          # 先占位
        sid = np.full(n, -1, dtype=np.int32)
        done = np.zeros(n, dtype=bool)

        k = np.searchsorted(bounds, start_abs, side="right")
        ov = (k < len(bounds)) & (bounds[np.minimum(k, len(bounds) - 1)] < end_abs)
        labels[ov] = 3
        done |= ov

        # 只遍历时间上可能相交的发作，避免每条记录都跑全患者发作
        lo = int(np.searchsorted(onsets, center.min() - (sop + sph) - 3600.0, side="left"))
        hi = int(np.searchsorted(onsets, center.max() + sop + sph, side="right"))
        for i in range(lo, hi):
            for mask, lab in (((onsets[i] <= center) & (center < offsets[i]), 2),
                              ((onsets[i] - sop - sph <= center) & (center < onsets[i] - sph), 1),
                              ((onsets[i] - sph <= center) & (center < onsets[i]), 3)):
                m = mask & ~done
                if m.any():
                    labels[m] = lab
                    sid[m] = i
                    done |= m

        labels[~done] = 4                             # 待定：由 INTER_THR 决定 0 / 3

        dprev = np.full(n, -1.0)
        m = pid >= 0
        dprev[m] = center[m] - offsets[pid[m]]
        dnext = np.full(n, -1.0)
        m = nid_ok
        dnext[m] = onsets[nid[m]] - center[m]

        parts.append((start_abs, center, prev_out, next_out, labels, sid, dprev, dnext))

    if not parts:
        return None
    return [np.concatenate([p[i] for p in parts]) for i in range(8)]


def apply_inter_thr(parts, inter_thr):
    """把待定窗 (base=4) 分成 0（间期）或 3（排除）"""
    start_abs, center, prev_out, next_out, base, sid, dprev, dnext = parts
    lab = base.copy()
    und = base == 4
    far_next = (next_out < 0) | (dnext > inter_thr)
    far_prev = (prev_out < 0) | (dprev > inter_thr)
    ok = und & far_next & far_prev
    lab[ok] = 0
    lab[und & ~ok] = 3
    return lab


def count_folds(lab, sid, prev_out, next_out, center, meta_cols=None):
    """用真实的 t.split_train_test + 两个过滤器（等价实现，按计数判）"""
    df = pd.DataFrame(dict(label=lab.astype(int), seizure_id=sid.astype(int),
                           prev_seizure_id=prev_out.astype(int),
                           next_seizure_id=next_out.astype(int),
                           center_abs_s=center.astype(float)))
    if (lab == 0).sum() == 0 or (lab == 1).sum() == 0:
        return 0, 0, 0, 0
    try:
        with io.StringIO() as _buf, contextlib.redirect_stdout(_buf):
            folds = t.split_train_test(df, seed=SEED, val=(ENC_VAL, CLS_VAL), method="LOOCV")
    except Exception:
        return 0, 0, 0, 0
    if not folds:
        return 0, 0, 0, 0

    ictal = lab == 2
    n_bal, n_final, min_train = 0, 0, 0
    for f in folds:
        if not all(f["n_%s_pre" % k] == f["n_%s_inter" % k] for k in ("train", "val", "test")):
            continue
        n_bal += 1
        ok = True
        sides = {"test": [f["seizure_id"]], "val": f["val_seizures"], "train": f["train_seizures"]}
        for key in ("test", "val", "train"):
            idx = f[key + "_idx"]
            # teacher_from_student 里 pos 是「这些发作的全部 ictal 窗」（不受 idx 限制），
            # neg 才是该边的间期窗，别写成 ictal[idx]
            pos = int(np.count_nonzero(ictal & np.isin(sid, sides[key])))
            neg = int(np.count_nonzero(lab[idx] == 0)) if len(idx) else 0
            if min(pos, neg) == 0:
                ok = False
                break
        if not ok:
            continue
        n_final += 1
        ntr = f["n_train_pre"] + f["n_train_inter"]
        min_train = ntr if min_train == 0 else min(min_train, ntr)
    return len(folds), n_bal, n_final, min_train


def scan_patient(pid, win_lens, overlaps, inter_thrs, elecs, sops, sphs):
    cache = __import__("json").loads((META / ("%s.json" % pid)).read_text(encoding="utf-8"))
    rows = []
    for wl in win_lens:
        for ov in overlaps:
            for sop in sops:
                for sph in sphs:
                    parts = build_windows(cache, wl, ov, elecs, sph=sph, sop=sop)
                    if parts is None:
                        continue
                    for ith in inter_thrs:
                        lab = apply_inter_thr(parts, ith * 3600.0)
                        fr, fb, ff, mtw = count_folds(lab, parts[5], parts[2], parts[3], parts[1])
                        rows.append(dict(patient=pid, win_len=wl, overlap=ov,
                                         sop_min=sop / 60.0, sph_min=sph / 60.0, inter_thr_h=ith,
                                         n_windows=int(len(lab)),
                                         usable=int(np.count_nonzero(np.isin(lab, (0, 1, 2)))),
                                         label0=int(np.count_nonzero(lab == 0)),
                                         label1=int(np.count_nonzero(lab == 1)),
                                         label2=int(np.count_nonzero(lab == 2)),
                                         label3=int(np.count_nonzero(lab == 3)),
                                         folds_raw=fr, folds_after_balance=fb, folds_final=ff,
                                         min_train_windows=mtw))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--stage1", action="store_true", help="只扫 SOP×SPH（4s/50%/2h）")
    ap.add_argument("--out", default="scan_results.csv")
    args = ap.parse_args()

    pids = sorted(p.stem for p in META.glob("chb*.json"))
    win_lens, overlaps, inter_thrs = WIN_LENS, OVERLAPS, INTER_THRS
    sops = [SOP]
    sphs = [SPH]
    if args.pilot:
        pids = ["chb16", "chb12", "chb01"]
        win_lens, overlaps, inter_thrs = [4, 8], [0.0, 0.5], [2.0, 3.5]
    if args.stage1:
        win_lens, overlaps, inter_thrs = [4], [0.5], [2.0]
        sops = [m * 60 for m in range(20, 91, 5)]      # SOP 20–90 min
        sphs = [m * 60 for m in range(2, 11)]          # SPH 2–10 min

    n_combo = len(win_lens) * len(overlaps) * len(inter_thrs) * len(sops) * len(sphs)
    print("患者 %d | 组合 %d" % (len(pids), n_combo))
    t0 = time.time()
    rows = []
    for i, pid in enumerate(pids, 1):
        rows += scan_patient(pid, win_lens, overlaps, inter_thrs, ELECS, sops, sphs)
        print("  [%d/%d] %s  累计 %.1f s" % (i, len(pids), pid, time.time() - t0), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / args.out, index=False, encoding="utf-8")
    print("写出", HERE / args.out, "行数", len(df), "用时 %.1f s" % (time.time() - t0))

    if len(df):
        agg = (df.groupby(["win_len", "overlap", "inter_thr_h"])
                 .agg(patients=("patient", "nunique"),
                      kept=("folds_final", lambda s: int((s >= MIN_FOLDS).sum())),
                      windows=("n_windows", "sum"))
                 .reset_index()
                 .sort_values(["kept", "windows"], ascending=False))
        print(agg.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
