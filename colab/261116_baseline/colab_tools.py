import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

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


def student_dataset(elec_names, meta, seed, val):
    """student_data_set[elec][fold]：只带索引和发作编号，不建张量
    （六电极索引相同；张量在每折里现建，避免所有折同时驻留内存）
    跟 261114 一样不丢折，折的选取交给主循环里 teacher_from_student 的 keep
    """
    raw = t.split_train_test(meta[elec_names[0]], seed=seed, val=val, method="LOOCV")
    keys = ("seizure_id", "val_seizures", "train_seizures", "train_idx", "val_idx", "test_idx")
    return [[{k: f[k] for k in keys} for f in raw] for _ in elec_names]


def classification_metrics(y_true, y_prob, threshold=0.5):
    """跟 tools.evaluate_classification 同口径（字段一致），只返回 dict 不打印"""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return dict(auc=float(roc_auc_score(y_true, y_prob)),
                accuracy=float(accuracy_score(y_true, y_pred)),
                balanced_acc=float((sensitivity + specificity) / 2),
                sensitivity=float(sensitivity),
                specificity=float(specificity),
                precision=float(precision),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def summarize_distill(out_root):
    """各患者的 {pid}.json -> 每患者每折一行：概率平均集成的 7 个指标"""
    rows = []
    for p in sorted(Path(out_root).glob("*/chb*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for e in data.get("test_index", []):
            pm = (e.get("metrics_pmean") or [{}])[0]
            rows.append(dict(
                patient=p.stem, fold=e["fold"], seizure_id=e.get("seizure_id"), alpha=e.get("alpha"),
                **{k + "_pmean": pm.get(k)
                   for k in ("auc", "accuracy", "balanced_acc", "sensitivity",
                             "specificity", "precision", "f1")},
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(out_root) / "summary_distill.csv", index=False, encoding="utf-8")
    return df
