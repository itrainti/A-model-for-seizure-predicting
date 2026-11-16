import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import tools as t

S3 = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"


def _retry(go, what, tries=4):
    for k in range(tries):
        try:
            return go()
        except Exception as e:
            if k == tries - 1:
                raise
            print("  [重试 %d/%d] %s: %s" % (k + 1, tries - 1, what, e), flush=True)
            time.sleep(2 ** k)


def fetch(rel, root):
    """rel 形如 'chb01/chb01_03.edf'；字节数对得上就跳过"""
    url = "%s/%s" % (S3, urllib.parse.quote(rel, safe="/"))
    dest = Path(root) / rel

    def head():
        req = urllib.request.Request(url, method="HEAD")
        return int(urllib.request.urlopen(req, timeout=30).headers["Content-Length"])

    want = _retry(head, "HEAD " + rel)
    if dest.exists() and dest.stat().st_size == want:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dest) + ".part")

    def go():
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if tmp.stat().st_size != want:
            raise RuntimeError("字节数不符 %d != %d" % (tmp.stat().st_size, want))
        os.replace(tmp, dest)
        return dest

    return _retry(go, rel)


def download_patient(pid, root):
    """下载 chbXX 的 summary 和它引用的全部 EDF，返回患者目录"""
    fetch("%s/%s-summary.txt" % (pid, pid), root)
    recs = t.read_summary(str(Path(root) / pid / ("%s-summary.txt" % pid)))
    files = sorted({r["file"] for r in recs})
    for i, f in enumerate(files):
        print("  [%s] %d/%d %s" % (pid, i + 1, len(files), f), flush=True)
        fetch("%s/%s" % (pid, f), root)
    return Path(root) / pid


def clear_patient(pid, root):
    shutil.rmtree(Path(root) / pid, ignore_errors=True)


def summarize(out_root):
    """把各患者的 {pid}.json 汇总成一张表：每患者每折的单电极均值 + 融合模型"""
    rows = []
    for p in sorted(Path(out_root).glob("*/chb*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        n_folds = len(data.get("weights", [])) // 6
        for e in (data.get("test_index") or [])[-n_folds:] if n_folds else []:
            m = e["metrics"]
            a = e["metrics_atten"][0] if e.get("metrics_atten") else {}
            rows.append(dict(
                patient=p.stem, fold=e["fold"],
                auc=float(np.mean([x["auc"] for x in m])),
                accuracy=float(np.mean([x["accuracy"] for x in m])),
                sensitivity=float(np.mean([x["sensitivity"] for x in m])),
                specificity=float(np.mean([x["specificity"] for x in m])),
                auc_atten=a.get("auc"),
                accuracy_atten=a.get("accuracy"),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_root) / "summary.csv", index=False, encoding="utf-8")
    return df


def summarize_teacher(out_root):
    """汇总各患者的教师指标：每折一行"""
    rows = []
    for p in sorted(Path(out_root).glob("*/chb*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for f in data.get("folds", []):
            rows.append(dict(patient=p.stem, fold=f["fold"], seizure_id=f["seizure_id"],
                             n_test=f["n_test"], auc=f["auc"], acc=f["acc"]))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_root) / "summary.csv", index=False, encoding="utf-8")
    return df
