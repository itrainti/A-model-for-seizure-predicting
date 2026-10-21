import argparse
import os

import numpy as np
import pandas as pd

import cache
import tools
from encoder import ConvEncoder, embed

SPH = 120
SOP = 1800
INTER_THR = 120 * 60
POST_THR = 180 * 60
WIN = 10
OVERLAP = 5
EMBED_LABELS = (0, 1, 2)


def patient_path(root, patient):
    return os.path.join(root, patient)


def dry_run(path, elec_names=None, n_sample=3, seed=0, inter_thr=INTER_THR, post_thr=POST_THR):
    elec_names = elec_names or tools.ELECS
    pid = os.path.basename(os.path.normpath(path))
    recs = tools.read_summary(tools._summary(path))
    seize = sorted(s for r in recs for s in r["seizures"])
    valid = tools.valid_seizures(recs, SPH, SOP, post_thr)

    print("Patient ID: %s" % pid)
    print("EDF 数: %d" % len(recs))
    print("seizure 总数: %d" % len(seize))
    print("有效 seizure 数: %d  %s" % (len(valid), valid))
    av = tools.channel_availability(path, elec_names)
    print("六通道可用: %s" % {c: bool(av[c]) for c in elec_names})
    for i, (a, b) in enumerate(seize):
        print("  S%-2d start=%.0f end=%.0f dur=%.0f  有效=%s"
              % (i, a, b, b - a, "Y" if i in valid else "N"))

    windows, meta = tools.preprocess(path, SPH, SOP, inter_thr, WIN, OVERLAP, elec_names)

    for c in elec_names:
        m = meta[c]
        print("[%s] %s 窗 %d usable %d" % (pid, c, len(m), int(m.usable.sum())))

    allm = pd.concat([meta[c] for c in elec_names], ignore_index=True)
    s = allm.sample(n=min(n_sample, len(allm)), random_state=seed)
    cols = ["edf_file", "channel", "start_abs_s", "center_abs_s", "next_seizure_id",
            "time_to_next_seizure_start_s", "label", "usable"]
    print("\n抽查窗口:")
    print(s[cols].to_string(index=False))

    pre = allm[allm.label == 1]
    if len(pre):
        print("\n首个 preictal 窗口整行:")
        for k, v in pre.iloc[0].items():
            print("  %-36s %s" % (k, v))
    return windows, meta


def build_patient(root, patient, out, encoder=None, elec_names=None, skip_existing=True,
                  inter_thr=INTER_THR, post_thr=POST_THR):
    elec_names = elec_names or tools.ELECS
    encoder = encoder or ConvEncoder()
    dim = encoder.dim
    windows, meta = tools.preprocess(patient_path(root, patient), SPH, SOP, inter_thr,
                                     WIN, OVERLAP, elec_names)
    for c in elec_names:
        d = os.path.join(out, c)
        h5 = os.path.join(d, "embeddings.h5")
        m = meta[c]
        if skip_existing and cache.exists(h5, dim):
            print("[跳过] %s 已存在 (%d 维)" % (h5, dim))
            continue
        sel = m[m.usable.eq(1) & m.label.isin(EMBED_LABELS)]
        row_id = sel.index.to_numpy(dtype=np.int64)
        emb = embed(windows[c][row_id], encoder)
        cache.save_metadata(os.path.join(d, "metadata.csv"), m)
        cache.save_embeddings(h5, emb, row_id,
                              dict(encoder=type(encoder).__name__, patient=patient, channel=c))
        print("[写出] %s  %d 窗 / embedding %s" % (d, len(m), emb.shape))


def verify(out, elec_names=None):
    elec_names = elec_names or tools.ELECS
    print("电极目录: %s" % sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))))
    for c in elec_names:
        d = os.path.join(out, c)
        m = cache.load_metadata(os.path.join(d, "metadata.csv"))
        emb, rid = cache.load_embeddings(os.path.join(d, "embeddings.h5"))
        assert emb.shape[0] == len(rid), c
        assert emb.dtype == np.float32, c
        assert rid.min() >= 0 and rid.max() < len(m), c
        assert np.all(np.diff(rid) > 0), c
        assert np.all(m.label.to_numpy()[rid] != 3), c
        assert np.all(m.usable.to_numpy()[rid] == 1), c
        assert not m.isna().any().any(), c
        assert set(m.channel.unique()) == {c}, c
        print("[校验] %-8s metadata %6d 行  embedding %s  row_id %d..%d"
              % (c, len(m), emb.shape, rid.min(), rid.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--patient")
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--val", type=float, default=None)
    ap.add_argument("--mode", default="expanding", choices=["expanding", "fixed"])
    ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--inter-thr", type=float, default=INTER_THR)
    ap.add_argument("--post-thr", type=float, default=POST_THR)
    a = ap.parse_args()

    if a.stats:
        tools.patient_statistics(a.root, tools.ELECS, interictal_thr=a.inter_thr, post_thr=a.post_thr)
        return
    if a.verify:
        verify(a.out)
        return
    if a.patient is None:
        ap.error("--patient 或 --stats / --verify 至少要给一个")
    if a.dry_run:
        windows, meta = dry_run(patient_path(a.root, a.patient),
                                inter_thr=a.inter_thr, post_thr=a.post_thr)
        m = meta[tools.ELECS[0]]
        for mode in ("expanding", "fixed"):
            tools.split_train_test(m, a.val, mode=mode, base=a.base)
        return
    build_patient(a.root, a.patient, a.out,
                  inter_thr=a.inter_thr, post_thr=a.post_thr)
    verify(a.out)


if __name__ == "__main__":
    main()
