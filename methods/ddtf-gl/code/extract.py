"""Compute and cache the features of every sample of a dataset (datasets.py order), once for all protocols, seeds and folds.

  python extract.py <dataset> [--features ddtf|dwt] [--workers 20] [--order P]

ddtf: dDTF maps (ddtf.py) with the settings calibrated on Safari et al.'s released maps (average reference where Source.ref says,
      sqrt(ffDTF * |pCoh|)); ddtf-v0: our first settings (ablation). Both: MVAR order chosen by AIC on up to 300 training samples of the dataset's first split (no labels used,
      protocols.order_pool): the median of the per-sample first local AIC minima over p = 1 .. ddtf.max_order(C, T)
dc:   control: the mean of each channel over the window (STEW-raw: the Emotiv DC level)
dwt:  the bundle's DWT statistics (upstream mwl_gl_si7.extract_dwt_features: Haar level 4, 9 statistics per channel and level, 16
      cross-channel summaries per level), on the raw windows, as the bundle's script computes them

Writes $GL_WORK/features/<dataset>/<features>/: X.npy (n, F) float32, meta.npz (labels, subjects, keys ...), info.json.
"""
import argparse
import json
import os
import sys
import time

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):      # one BLAS thread per worker process
    os.environ[k] = '1'
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import ddtf  # noqa: E402
import protocols  # noqa: E402

WORK = os.path.expanduser(os.environ.get('GL_WORK', '~/repro-gl'))
UPSTREAM = os.environ.get('GL_UPSTREAM', os.path.join(HERE, '../upstream'))
SRC, ORDER, KIND = None, None, None


def work(idx):
    x = SRC.window(idx)
    if KIND == 'ddtf':                                   # the settings calibrated on Safari et al.'s maps
        return ddtf.band_ddtf(x, ORDER, SRC.fs, ref=SRC.ref, form='safari').reshape(len(idx), -1)
    if KIND == 'ddtf-v0':                                # our first settings (ablation)
        return ddtf.band_ddtf(x, ORDER, SRC.fs, form='v0').reshape(len(idx), -1)
    if KIND == 'dc':                                     # control: each channel's mean over the window (its DC level)
        return x.mean(-1).astype(np.float32)
    sys.path.insert(0, UPSTREAM)
    from mwl_gl_si7 import extract_dwt_features
    return extract_dwt_features(x.astype(np.float32), 'haar', 4, 8, progress=False)


def main():
    global SRC, ORDER, KIND
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--features', default='ddtf', choices=['ddtf', 'ddtf-v0', 'dwt', 'dc'])
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--order', type=int, default=0, help='MVAR order; 0 = choose by AIC')
    a = ap.parse_args()
    KIND = a.features
    out = os.path.join(WORK, 'features', a.dataset, a.features)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    SRC = datasets.get(a.dataset)
    meta = {'y': SRC.y}
    for k in ('subject', 'stage', 'rating', 'rec'):
        if hasattr(SRC, k):
            meta[k] = getattr(SRC, k)
    if hasattr(SRC, 'keys'):
        meta['keys'] = np.array(SRC.keys)
        for p in ('train', 'val', 'test'):
            meta[f'n_{p}'] = len(SRC.split_keys[p])
    np.savez(os.path.join(out, 'meta.npz'), **meta)
    if a.dataset in ('SEED', 'SEED-V'):                  # both splits' key lists, so runs need no LMDB
        json.dump({sp: datasets.lmdb_split_keys(a.dataset, sp) for sp in ('trial555', 'subject')}, open(os.path.join(out, 'split_keys.json'), 'w'))
    x0 = SRC.window([0])
    close = getattr(SRC, 'close', lambda: None)    # py-lmdb: one open environment per path and process (protocols reopens SEED's)
    close()
    C, T = x0.shape[1:]
    info = dict(dataset=a.dataset, features=a.features, n=len(SRC), channels=SRC.map_channels(), fs=SRC.fs, samples=int(T),
                dropped_last_channel=SRC.drop_last, load_seconds=round(time.time() - t0, 1))
    print(f'{a.dataset}: {len(SRC)} samples, {C} channels x {T} samples at {SRC.fs} Hz (loaded in {info["load_seconds"]} s)', flush=True)
    if a.features in ('ddtf', 'ddtf-v0'):
        info.update(form='safari' if a.features == 'ddtf' else 'v0',
                    average_reference=[SRC.map_channels()[i] for i in SRC.ref] if (a.features == 'ddtf' and SRC.ref is not None) else None)
        pool = protocols.order_pool(a.dataset, meta)
        pick = np.sort(np.random.default_rng(0).choice(pool, size=min(300, len(pool)), replace=False))
        pmax = ddtf.max_order(C, T)
        if a.order:
            ORDER = a.order
        else:
            t1 = time.time()
            xs = SRC.window(pick)
            close()
            ORDER, best, _ = ddtf.select_order(xs, pmax)
            info.update(order_by='AIC: first local minimum per sample, median over samples', order_candidates=[1, pmax],
                        order_histogram=np.bincount(best, minlength=pmax + 1)[1:].tolist(), order_seconds=round(time.time() - t1, 1))
        info.update(order=ORDER, bands=[list(b) for b in ddtf.BANDS], freq_grid_hz=[1, int(SRC.fs // 2)])
        print(f'MVAR order {ORDER} (AIC over p = 1..{pmax}: {info.get("order_histogram")})', flush=True)
        chunk = min(256, ddtf.chunk_size(C, SRC.fs))
    elif a.features == 'dwt':
        info.update(dwt=dict(wavelet='haar', level=4, corr_top_k=8))
        chunk = 64
    else:
        info.update(dc='mean of each channel over the window')
        chunk = 256
    ORDER = int(ORDER or 0)
    F = work([0, 1]).shape[1]
    close()
    X = np.lib.format.open_memmap(os.path.join(out, 'X.npy.part'), mode='w+', dtype=np.float32, shape=(len(SRC), F))
    chunks = [np.arange(s, min(s + chunk, len(SRC))) for s in range(0, len(SRC), chunk)]
    t1 = time.time()
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers) as p:
        for i, (idx, f) in enumerate(zip(chunks, p.imap(work, chunks))):
            X[idx] = f
            if i % max(1, len(chunks) // 20) == 0:
                print(f'  {idx[-1] + 1}/{len(SRC)} samples, {time.time() - t1:.0f} s', flush=True)
    X.flush()
    bad = int((~np.isfinite(X)).any(1).sum())
    if bad:
        print(f'WARNING: {bad} samples with non-finite features, set to 0', flush=True)
        for s in range(0, len(X), 4096):
            X[s:s + 4096] = np.nan_to_num(X[s:s + 4096], nan=0.0, posinf=0.0, neginf=0.0)
        X.flush()
    del X
    os.replace(os.path.join(out, 'X.npy.part'), os.path.join(out, 'X.npy'))
    info.update(n_features=F, nonfinite_samples=bad, extract_seconds=round(time.time() - t1, 1), workers=a.workers,
                cpu_ms_per_sample=round(1000 * (time.time() - t1) * a.workers / len(SRC), 2))
    json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'channels'}), flush=True)


if __name__ == '__main__':
    main()
