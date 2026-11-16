"""SOP × (窗长, 重叠, 间期阈值) 交叉扫描，SPH 固定 2 min。

组合数 = 12 × 4 × 7 × 15 = 5040，按患者多进程并行。

用法：
    python scan_cross.py --workers 8
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

import scan

ELECS = scan.ELECS
SPH_FIXED = 2 * 60
HERE = Path(__file__).resolve().parent


def run_one(pid):
    rows = scan.scan_patient(
        pid,
        win_lens=scan.WIN_LENS,                 # 4..15
        overlaps=scan.OVERLAPS,                 # 0/25/50/75%
        inter_thrs=scan.INTER_THRS,             # 2.0..5.0 h
        elecs=ELECS,
        sops=[m * 60 for m in range(20, 91, 5)],
        sphs=[SPH_FIXED],
    )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="scan_cross.csv")
    args = ap.parse_args()

    pids = sorted(p.stem for p in scan.META.glob("chb*.json"))
    print("患者 %d | 组合 %d | workers %d"
          % (len(pids), len(scan.WIN_LENS) * len(scan.OVERLAPS) * len(scan.INTER_THRS) * 15,
             args.workers), flush=True)

    t0 = time.time()
    rows = []
    with mp.Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(run_one, pids), 1):
            rows += part
            print("  [%d/%d] 累计 %.1f s" % (i, len(pids), time.time() - t0), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / args.out, index=False, encoding="utf-8")
    print("写出 %s 行数 %d 用时 %.1f s" % (HERE / args.out, len(df), time.time() - t0))


if __name__ == "__main__":
    main()
