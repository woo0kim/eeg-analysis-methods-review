"""Every number of a dDTF-GL experiment from its runs (logs/runs/<run>/{metrics.json, test_preds.npz}); no data or GPU needed.

  python collect.py <experiment> <runs dir> <results dir> > <results dir>/results.txt

Writes <results dir>/results.json in the schema of the pipeline the experiment joins, so tools/compare_*.py read dDTF-GL next to the
other models without changes to their loaders, and <results dir>/summary.csv (docs/adding-an-experiment.md).

  stew-cbramod-pipeline      as finetune-stew/code/collect.py: per seed, the 7 test folds pooled (they cover every window once)
  stew-paper-protocol        the paper's Tables 3-4 metrics (accuracy, macro F1 / precision / recall), mean over folds and SD over folds,
                             as the paper reports them; the bundle's own runs from their mwl_gl_si7.json
  cbramod-splits             as the CBraMod-pipeline collectors: balanced accuracy, kappa, weighted F1 (multi-class) or balanced
                             accuracy, AUC-PR, AUROC (binary)
  sleepedf-cbramod-pipeline  as finetune-sleepedf/code/collect.py, including the common test set (results/test_epochs.json there)
  seed-cbramod-pipeline      as finetune-seed/code/collect.py, including the clip vote (results/test_trials.json there)
"""
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, cohen_kappa_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
CBRAMOD = os.path.normpath(os.path.join(HERE, '../../cbramod/experiments'))
LABEL = {'ddtf-gl': 'dDTF-GL', 'ddtf-gl-noaxb': 'dDTF-GL, no AXB', 'dwt-gl': 'DWT-GL (bundle features), no AXB', 'dwt-gl-axb': 'DWT-GL (bundle features) + AXB',
         'ddtf-gl-v0': 'dDTF-GL, first map settings', 'ddtf-gl-noaxb-v0': 'dDTF-GL, first map settings, no AXB', 'safari-gl': "dDTF-GL on Safari et al.'s maps",
         'safari-gl-gpu': "dDTF-GL (GPU) on Safari et al.'s maps"}
SETTING = {'ddtf-gl': "our implementation of the paper's dDTF -> DFT -> AXB -> LNT -> XGBoost",
           'ddtf-gl-noaxb': 'dDTF -> DFT -> LNT -> XGBoost (no auxiliary XGBoost)',
           'dwt-gl': "the authors' bundle chain (DWT statistics -> DFT -> LNT -> XGBoost) on the shared windows and folds",
           'dwt-gl-axb': 'DWT statistics -> DFT -> AXB -> LNT -> XGBoost',
           'ddtf-gl-v0': "the paper's chain on dDTF maps with our first settings (no re-reference, ffDTF x squared pCoh)",
           'ddtf-gl-noaxb-v0': 'dDTF maps with our first settings -> DFT -> LNT -> XGBoost',
           'safari-gl': "the paper's DFT -> AXB -> LNT -> XGBoost on Safari et al.'s released SIFT dDTF maps",
           'safari-gl-gpu': "the same with XGBoost (AXB and classifier) on the GPU, the paper's GPU variant"}
METHOD = 'dDTF-GL'
PAPER = {  # Cao, Li & Kuo, Tables 3 (SD) and 4 (SI), "Our Method (CPU)": accuracy, F1, precision, recall, mean and SD
    'sd': {'accuracy': (0.898, 0.016), 'f1': (0.898, 0.016), 'precision': (0.899, 0.015), 'recall': (0.898, 0.015)},
    'si': {'accuracy': (0.766, 0.148), 'f1': (0.756, 0.161), 'precision': (0.781, 0.148), 'recall': (0.766, 0.148)}}
PAPER_GPU = {  # the same tables, "Our Method (GPU)"
    'sd': {'accuracy': (0.896, 0.017), 'f1': (0.896, 0.018), 'precision': (0.896, 0.017), 'recall': (0.896, 0.017)},
    'si': {'accuracy': (0.765, 0.149), 'f1': (0.756, 0.163), 'precision': (0.780, 0.150), 'recall': (0.765, 0.149)}}


def runs(path):
    out = {}
    for d in sorted(glob.glob(os.path.join(path, '*'))):
        m, p = os.path.join(d, 'metrics.json'), os.path.join(d, 'test_preds.npz')
        if os.path.exists(m) and os.path.exists(p):
            out[os.path.basename(d)] = (json.load(open(m)), np.load(p))
    return out


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def write_csv(rows, results):
    with open(os.path.join(results, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'dataset', 'protocol', 'setting', 'metric', 'paper_mean', 'paper_std', 'paper_source',
                                          'ours_mean', 'ours_std', 'n_runs', 'headline'])
        w.writeheader()
        w.writerows(rows)


def row(dataset, protocol, setting, metric, mean, std, n, headline, paper=None, source=''):
    return dict(method=METHOD, dataset=dataset, protocol=protocol, setting=setting, metric=metric,
                paper_mean=paper[0] if paper else '', paper_std=paper[1] if paper else '', paper_source=source if paper else '',
                ours_mean=round(float(mean), 4), ours_std=round(float(std), 4) if n > 1 else '', n_runs=n, headline=int(headline))


def costs(rs):
    """Mean feature counts per stage, model size and timings over runs."""
    keys = ['input', 'dft', 'axb', 'lnt', 'final']
    nf = {k: float(np.mean([r['n_features'][k] for r in rs if k in r['n_features']])) for k in keys if any(k in r['n_features'] for r in rs)}
    return dict(n_features=nf, trees=float(np.mean([r['model']['trees'] for r in rs])), nodes=float(np.mean([r['model']['nodes'] for r in rs])),
                fit_seconds=float(np.mean([r['seconds']['fit_total'] for r in rs])),
                predict_ms_per_sample=float(np.mean([r['seconds']['predict_ms_per_sample'] for r in rs])))


def fmt_costs(c):
    nf = ' -> '.join(f'{k} {v:.0f}' for k, v in c['n_features'].items())
    return f'features {nf}; {c["trees"]:.0f} trees, {c["nodes"]:.0f} nodes; fit {c["fit_seconds"]:.0f} s; predict {c["predict_ms_per_sample"]:.4f} ms/sample'


# ---------------------------------------------------------------------------------------------------------------- STEW, shared folds
def stew(R, OUT):
    names = ['accuracy', 'balanced_acc', 'kappa', 'weighted_f1', 'macro_f1']
    sc = lambda y, p: [accuracy_score(y, p), balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'),
                       f1_score(y, p, average='macro')]
    data = {}
    for name, (m, z) in R.items():
        task, fold, model, seed = name.split('__')
        data.setdefault((task, model, int(seed[1:])), {})[int(fold[4:])] = (z['y'].astype(int), z['pred'].astype(int), m)
    rows, summary, all_m = [], [], {}
    for (task, model, seed), folds in sorted(data.items()):
        all_m.setdefault((task, model), []).extend(folds[f][2] for f in folds)
        if len(folds) < 7:
            print('incomplete', task, model, seed, sorted(folds))
            continue
        y, p = np.concatenate([folds[k][0] for k in range(7)]), np.concatenate([folds[k][1] for k in range(7)])
        k = int(y.max() + 1)
        rows.append(dict(task=task, model=model, seed=seed, metric_names=names, metrics=sc(y, p), per_fold=[sc(*folds[f][:2]) for f in range(7)],
                         val=[folds[f][2]['val'] for f in range(7)], cm=np.bincount(k * y + p, minlength=k * k).reshape(k, k).tolist(),
                         costs=costs([folds[f][2] for f in range(7)])))
    for key in sorted({(r['task'], r['model']) for r in rows}):
        rs = [r for r in rows if (r['task'], r['model']) == key]
        a, f = np.array([r['metrics'] for r in rs]), np.array([r['per_fold'] for r in rs])
        summary.append(dict(task=key[0], model=key[1], n_seeds=len(rs), metric_names=names, mean=a.mean(0).tolist(),
                            std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(5)).tolist(), fold_mean=f.mean((0, 1)).tolist(),
                            fold_sd=f.std(1, ddof=1).mean(0).tolist(), costs=costs(all_m[key])))
        s = summary[-1]
        print(f'{key[0]:10s} {key[1]:17s} seeds={len(rs)} ' + ' '.join(f'{n} {m:.4f}±{v:.4f}' for n, m, v in zip(names, s['mean'], s['std']))
              + f'  fold SD(acc) {s["fold_sd"][0]:.4f}')
    print()
    for s in summary:
        print(f'{s["task"]:10s} {s["model"]:17s} {fmt_costs(s["costs"])}')
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    PROT = {'rest-task': 'cross-subject GroupKFold(7), rest vs. task', 'rating3': 'cross-subject GroupKFold(7), 3 rating levels'}
    write_csv([row('STEW', PROT[s['task']], SETTING[s['model']] + ', 4-s windows, shared folds', n, s['mean'][i], s['std'][i], s['n_seeds'] * 7,
                   n == 'balanced_acc' and s['model'] == 'ddtf-gl')
               for s in summary for i, n in enumerate(s['metric_names']) if n != 'macro_f1'], OUT)


# ------------------------------------------------------------------------------------------------------- STEW, the paper's protocol
def paper_metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), f1=f1_score(y, p, average='macro'), precision=precision_score(y, p, average='macro', zero_division=0),
                recall=recall_score(y, p, average='macro'))


def stew_paper(R, OUT, bundle_dir):
    data = {}
    for name, (m, z) in R.items():
        prot, fold, model, seed = name.split('__')
        data.setdefault((prot, model, int(seed[1:])), {})[int(fold[4:])] = (z['y'].astype(int), z['pred'].astype(int), m)
    out, rows = {'runs': [], 'summary': []}, []
    names = ['accuracy', 'f1', 'precision', 'recall']
    for (prot, model, seed), folds in sorted(data.items()):
        if len(folds) < 7:
            print('incomplete', prot, model, seed, sorted(folds))
            continue
        per = [paper_metrics(*folds[f][:2]) for f in range(7)]
        y, p = np.concatenate([folds[f][0] for f in range(7)]), np.concatenate([folds[f][1] for f in range(7)])
        out['runs'].append(dict(protocol=prot, model=model, seed=seed, per_fold=per, pooled=paper_metrics(y, p),
                                fold_mean={n: float(np.mean([q[n] for q in per])) for n in names},
                                fold_sd={n: float(np.std([q[n] for q in per], ddof=1)) for n in names}, costs=costs([folds[f][2] for f in range(7)])))
    print("STEW, the paper's protocol (6-s windows, 4-s step). Mean over folds; ± = SD over the 7 folds (the paper's ±), averaged over seeds;")
    print('[seeds] = SD over seeds of the fold mean. Paper: Cao et al., Tables 3 (SD) and 4 (SI), "Our Method (CPU)".\n')
    for prot, variant in itertools.product(('si', 'sd', 'sd-blocked'), ('safari-gl', 'safari-gl-gpu', 'ddtf-gl', 'ddtf-gl-v0')):
        rs = [r for r in out['runs'] if r['protocol'] == prot and r['model'] == variant]
        if not rs:
            continue
        s = dict(protocol=prot, model=rs[0]['model'], n_seeds=len(rs))
        line = []
        for n in names:
            fm = [r['fold_mean'][n] for r in rs]
            s[n] = dict(mean=float(np.mean(fm)), seed_sd=float(np.std(fm, ddof=1)) if len(fm) > 1 else 0.0,
                        fold_sd=float(np.mean([r['fold_sd'][n] for r in rs])))
            pm, psd = (PAPER_GPU if variant.endswith('-gpu') else PAPER)['sd' if prot == 'sd-blocked' else prot][n]
            line.append(f'{n} {s[n]["mean"]:.4f} ± {s[n]["fold_sd"]:.4f} [seeds {s[n]["seed_sd"]:.4f}] (paper {pm:.3f} ± {psd:.3f}, Δ {s[n]["mean"] - pm:+.4f})')
        out['summary'].append(s)
        print(f'{prot.upper()} {LABEL[variant]}, {len(rs)} seeds x 7 folds: '
              + '; '.join(line).replace('(paper', '(paper SD' if prot == 'sd-blocked' else '(paper'))
        print(f'   {fmt_costs(rs[0]["costs"])}')
        if prot == 'si':   # SD over the 48 subjects, the other reading of the paper's large SD (test order: subjects ascending, 74 windows each)
            test_s = json.load(open(os.path.join(CBRAMOD, 'finetune-stew/results/folds.json')))['rest-task']['test_subjects']
            sd_subj = []
            for (pp, model, seed_), folds in data.items():
                if pp == 'si' and model == variant and len(folds) == 7:
                    acc = [np.mean(folds[f][0][subj == s_] == folds[f][1][subj == s_]) for f in range(7)
                           for subj in [np.repeat(sorted(test_s[f]), 74)] for s_ in sorted(test_s[f])]
                    sd_subj.append((np.mean(acc), np.std(acc, ddof=1)))
            s['subject_accuracy'] = dict(mean=float(np.mean([a for a, _ in sd_subj])), sd=float(np.mean([b for _, b in sd_subj])))
            print(f'   per subject: accuracy {s["subject_accuracy"]["mean"]:.4f} ± {s["subject_accuracy"]["sd"]:.4f} (SD over the 48 subjects)')
            prot_label = 'subject-independent GroupKFold(7), 6-s windows (paper)'
        elif prot == 'sd':
            prot_label = 'subject-dependent, 7-fold CV over pooled 6-s windows (paper)'
        else:
            prot_label = 'subject-dependent, 7 contiguous blocks per recording, no overlapping windows across the split'
        setting = SETTING[variant] + ("; the paper's windows" if variant == 'safari-gl' else "; the paper's windows, 1-Hz high-pass")
        for n in names:
            rows.append(row('STEW', prot_label, setting + '; mean over folds, SD over folds', n,
                            s[n]['mean'], s[n]['fold_sd'], len(rs) * 7, n == 'accuracy' and prot != 'sd-blocked' and variant == 'safari-gl',
                            (PAPER_GPU if variant.endswith('-gpu') else PAPER)['sd' if prot == 'sd-blocked' else prot][n],
                            f'Cao et al., Table {4 if prot == "si" else 3}' + (', GPU row' if variant.endswith('-gpu') else '')))
    # the paper's GPU variant against the CPU one (same maps, splits and seeds; only XGBoost's device differs)
    from scipy import stats as _st
    for prot in ('si', 'sd', 'sd-blocked'):
        c = [r['fold_mean']['accuracy'] for r in out['runs'] if r['protocol'] == prot and r['model'] == 'safari-gl']
        g = [r['fold_mean']['accuracy'] for r in out['runs'] if r['protocol'] == prot and r['model'] == 'safari-gl-gpu']
        if len(c) > 1 and len(g) > 1:
            p_ = _st.ttest_ind(g, c, equal_var=False).pvalue
            print(f'{prot.upper()}: GPU - CPU (Safari et al.s maps), accuracy {np.mean(g) - np.mean(c):+.4f} (Welch p {p_:.3f}, seeds {len(g)} / {len(c)})')
    # controls on the bundle's 4-s windows and folds
    ctrl = sorted({(r['protocol'], r['model']) for r in out['runs'] if r['protocol'].startswith('si4')})
    if ctrl:
        print("\nControls on the bundle's 4-s windows (25% overlap) and folds, training on six folds; si4raw = unfiltered, as the bundle reads them,"
              "\nsi4hp = high-passed at 0.3 Hz (finetune-stew's windows). dc-gl = only each channel's mean. Window accuracy, mean over folds ± SD over folds:")
        for prot, model in ctrl:
            rs = [r for r in out['runs'] if (r['protocol'], r['model']) == (prot, model)]
            fm = [r['fold_mean']['accuracy'] for r in rs]
            c = dict(protocol=prot, model=model, n_seeds=len(rs), accuracy=dict(mean=float(np.mean(fm)), fold_sd=float(np.mean([r['fold_sd']['accuracy'] for r in rs])),
                     seed_sd=float(np.std(fm, ddof=1)) if len(fm) > 1 else 0.0), per_seed=fm)
            out['summary'].append(c)
            print(f'  {prot:7s} {model:8s} {len(rs)} seeds: {c["accuracy"]["mean"]:.4f} ± {c["accuracy"]["fold_sd"]:.4f} [seeds {c["accuracy"]["seed_sd"]:.4f}]; '
                  f'{fmt_costs(rs[0]["costs"])}')
    # the authors' bundle as released
    b = []
    for d in sorted(glob.glob(os.path.join(bundle_dir, 'bundle__s*'))):
        j = os.path.join(d, 'mwl_gl_si7.json')
        if os.path.exists(j):
            r = json.load(open(j))
            if r.get('complete'):
                b.append(dict(seed=r['config']['seed'], summary=r['summary'], folds=[dict(fold=f['fold'], test_subjects=f['test_subjects'],
                         segment=f['segment'], trial=f['trial'], n_dft=f['n_dft'], n_selected=f['n_selected']) for f in r['folds']]))
    if b:
        print(f"\nThe authors' bundle as released (DWT -> DFT -> LNT -> XGBoost, 4-s windows with 25% overlap, its own GroupKFold(7)), {len(b)} seeds:")
        out['bundle'] = b
        for key, lab in (('segment_acc', 'window accuracy'), ('segment_f1_macro', 'window macro F1'), ('trial_acc', 'recording vote accuracy')):
            m = [x['summary'][key]['mean'] for x in b]
            sd = [x['summary'][key]['std'] for x in b]       # np.std over folds (ddof 0), as the bundle computes it
            print(f'  {lab}: {np.mean(m):.4f} ± {np.mean(sd):.4f} (SD over folds, ddof 0) [seeds {np.std(m, ddof=1) if len(m) > 1 else 0:.4f}]; per seed '
                  + ', '.join(f'{v:.4f}' for v in m))
        print(f'  test subjects of fold 0: {b[0]["folds"][0]["test_subjects"]}')
        bs = 'the authors\' bundle as released: DWT statistics -> DFT -> LNT -> XGBoost, 4-s windows (25% overlap), its own GroupKFold(7); mean and SD over folds'
        for key, n in (('segment_acc', 'accuracy'), ('segment_f1_macro', 'f1')):
            rows.append(row('STEW', 'subject-independent GroupKFold(7), the bundle\'s own windows and folds', bs, n,
                            np.mean([x['summary'][key]['mean'] for x in b]), np.mean([x['summary'][key]['std'] for x in b]), len(b) * 7, False,
                            PAPER['si'][n], 'Cao et al., Table 4'))
    nets = networks(os.path.join(os.path.dirname(os.path.normpath(bundle_dir)), 'runs_networks'), data, out, rows)
    if nets:
        out['networks'] = nets
    json.dump(out, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    write_csv(rows, OUT)


GL_REF = 'safari-gl'   # the networks are tested against dDTF-GL on Safari et al.'s maps, the paper's own input
NETS = [('eegnet', 'EEGNet', 'EEGNet', 'our PyTorch EEGNet-8,2'), ('conformer', 'EEG Conformer', 'EEG-Conformer', 'released model'),
        ('bd-atcnet', 'ATCNet (bd)', 'ATCNet', 'braindecode'), ('bd-deepsleepnet', 'DeepSleepNet (bd)', 'DeepSleepNet', 'braindecode'),
        ('dgcnn', 'DGCNN', 'DGCNN', 'ours, 5-band DE'), ('cbramod', 'CBraMod', 'CBraMod', 'released code and weights'),
        ('labram', 'LaBraM-Base', 'LaBraM', 'released code and weights, its recipe')]


def networks(path, gl_data, out, rows):
    """The repository's networks in the paper's two settings (code/nets6.py, run_labram6.py; the final epoch of 50, no selection),
    scored as dDTF-GL above, with Welch tests of dDTF-GL against each (fold-mean accuracy per seed)."""
    from scipy import stats
    if not os.path.isdir(path):
        return None
    data = {}
    for d in sorted(glob.glob(os.path.join(path, '*__fold*__*__s*'))):
        prot, fold, model, seed = os.path.basename(d).split('__')
        t, p = os.path.join(d, 'test_preds.npz'), os.path.join(d, 'preds.npz')
        if os.path.exists(t):
            z = np.load(t)
            y, pr = z['y'].astype(int), z['pred'].astype(int)
        elif os.path.exists(p):                                   # LaBraM: probabilities of every epoch; the final one is scored
            try:
                z = np.load(p)
                pt = z['test']
            except Exception:
                continue
            if len(pt) < 50:
                continue
            y, pr = z['y_test'].astype(int), pt[-1].argmax(-1)
            # as the LaBraM collectors elsewhere: the probabilities are too large to commit; keep the final epoch's predictions and
            # every epoch's test scores (recorded, never used)
            np.savez_compressed(t, y=y.astype(np.int8), pred=pr.astype(np.int8), epoch=len(pt))
            json.dump([{'epoch': e + 1, 'test': {'accuracy': float(np.mean(pt[e].argmax(-1) == y)), 'kappa': float(cohen_kappa_score(y, pt[e].argmax(-1)))}}
                       for e in range(len(pt))], open(os.path.join(d, 'metrics.json'), 'w'))
        else:
            continue
        data.setdefault((prot, model, int(seed[1:])), {})[int(fold[4:])] = (y, pr)
    if not data:
        return None
    test_s = json.load(open(os.path.join(CBRAMOD, 'finetune-stew/results/folds.json')))['rest-task']['test_subjects']
    names = ['accuracy', 'f1', 'precision', 'recall']
    res = {}
    for (prot, model, seed), folds in sorted(data.items()):
        if len(folds) < 7:
            print('incomplete', prot, model, seed, sorted(folds))
            continue
        per = [paper_metrics(*folds[f]) for f in range(7)]
        r = dict(seed=seed, fold_mean={n: float(np.mean([q[n] for q in per])) for n in names},
                 fold_sd={n: float(np.std([q[n] for q in per], ddof=1)) for n in names})
        if prot == 'si':
            acc = [np.mean(folds[f][0][subj == s_] == folds[f][1][subj == s_]) for f in range(7)
                   for subj in [np.repeat(sorted(test_s[f]), 74)] for s_ in sorted(test_s[f])]
            r['subject_sd'] = float(np.std(acc, ddof=1))
        res.setdefault(prot, {}).setdefault(model, []).append(r)
    gl = {}
    for (prot, model, seed), folds in gl_data.items():
        if model == GL_REF and prot in ('si', 'sd') and len(folds) == 7:
            gl.setdefault(prot, []).append(np.mean([paper_metrics(*folds[f][:2])['accuracy'] for f in range(7)]))
    print("\nThe repository's networks in the paper's settings (same windows and splits; 50 epochs, the final one scored, no validation).")
    print('Accuracy, mean over folds ± SD over folds [SD over seeds]; SI also SD over subjects; Welch p of dDTF-GL against each, over seeds:')
    summary = {}
    for prot in ('si', 'sd'):
        print(f'\n{prot.upper()}   dDTF-GL {np.mean(gl.get(prot, [np.nan])):.4f}')
        for key, label, method, impl in NETS:
            rs = res.get(prot, {}).get(key)
            if not rs:
                continue
            fm = [r['fold_mean']['accuracy'] for r in rs]
            s = {n: dict(mean=float(np.mean([r['fold_mean'][n] for r in rs])), fold_sd=float(np.mean([r['fold_sd'][n] for r in rs])),
                         seed_sd=float(np.std([r['fold_mean'][n] for r in rs], ddof=1)) if len(rs) > 1 else 0.0) for n in names}
            s['n_seeds'] = len(rs)
            if prot == 'si':
                s['subject_sd'] = float(np.mean([r['subject_sd'] for r in rs]))
            p = stats.ttest_ind(gl[prot], fm, equal_var=False).pvalue if len(gl.get(prot, [])) > 1 and len(fm) > 1 else float('nan')
            s['ddtf_gl_minus'] = dict(diff=float(np.mean(gl[prot]) - np.mean(fm)), p=float(p))
            summary.setdefault(prot, {})[key] = s
            print(f'  {label:18s} {len(rs)} seeds: accuracy {s["accuracy"]["mean"]:.4f} ± {s["accuracy"]["fold_sd"]:.4f} [{s["accuracy"]["seed_sd"]:.4f}]'
                  + (f' (subjects ± {s["subject_sd"]:.4f})' if prot == 'si' else '') + f'; F1 {s["f1"]["mean"]:.4f}; precision {s["precision"]["mean"]:.4f}'
                  f'; recall {s["recall"]["mean"]:.4f}; dDTF-GL − it {s["ddtf_gl_minus"]["diff"]:+.4f} (p {"< 0.001" if p < 0.001 else f"= {p:.3f}"})')
            label_p = ('subject-independent GroupKFold(7), 6-s windows (paper)' if prot == 'si'
                       else 'subject-dependent, 7-fold CV over pooled 6-s windows (paper)')
            for n in names:
                rows.append(dict(method=method, dataset='STEW', protocol=label_p, setting=f"{impl}, the dDTF-GL paper's windows and splits, "
                                 '50 epochs, final epoch; mean over folds, SD over folds', metric=n, paper_mean='', paper_std='', paper_source='',
                                 ours_mean=round(s[n]['mean'], 4), ours_std=round(s[n]['fold_sd'], 4), n_runs=len(rs) * 7, headline=0))
    return dict(runs=res, summary=summary)


# ------------------------------------------------------------------------------------------------------------- CBraMod's four datasets
DATASET = {'PhysioNet-MI': ('PhysioNet-MI', 'cross-subject 70/19/20'), 'BCIC-IV-2a': ('BCI-IV-2a', 'cross-subject 5/2/2'),
           'Mumtaz2016': ('Mumtaz2016', 'cross-subject 43/9/11'), 'MentalArithmetic': ('MentalArithmetic', 'cross-subject 28/4/4')}


def splits(R, OUT):
    rows = []
    for name, (m, z) in R.items():
        ds, model, seed = name.split('__')
        binary = 'roc_auc' in m['test']
        names = ['balanced_acc', 'pr_auc', 'roc_auc'] if binary else ['balanced_acc', 'kappa', 'weighted_f1']
        rows.append(dict(dataset=ds, setting=model, seed=int(seed[1:]), metric_names=names, metrics=[m['test'][k] for k in names],
                         val_metrics=[m['val'][k] for k in names], accuracy=m['test']['accuracy'], costs=costs([m]),
                         mvar_order=m['features'].get('order')))
    summary = []
    for key in sorted({(r['dataset'], r['setting']) for r in rows}):
        rs = sorted((r for r in rows if (r['dataset'], r['setting']) == key), key=lambda r: r['seed'])
        a = np.array([r['metrics'] for r in rs])
        summary.append(dict(dataset=key[0], setting=key[1], n=len(rs), seeds=[r['seed'] for r in rs], metric_names=rs[0]['metric_names'],
                            per_seed=a.tolist(), mean=a.mean(0).tolist(), std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(3)).tolist(),
                            costs=rs[0]['costs'], mvar_order=rs[0]['mvar_order']))
        s = summary[-1]
        print(f"{key[0]:17s} {key[1]:12s} n={len(rs)} " + '  '.join(f'{n} {mm:.4f} ± {v:.4f}' for n, mm, v in zip(s['metric_names'], s['mean'], s['std'])))
    print()
    for s in summary:
        print(f"{s['dataset']:17s} MVAR order {s['mvar_order']}; {fmt_costs(s['costs'])}")
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    write_csv([row(*DATASET[s['dataset']], SETTING[s['setting']] + ", on CBraMod's preprocessed samples and splits", n, s['mean'][i], s['std'][i], s['n'],
                   n == 'balanced_acc' and s['setting'] == 'ddtf-gl') for s in summary for i, n in enumerate(s['metric_names'])], OUT)


# ------------------------------------------------------------------------------------------------------------------------- Sleep-EDF
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def sleep(R, OUT):
    te = json.load(open(os.path.join(CBRAMOD, 'finetune-sleepedf/results/test_epochs.json')))
    labels, mask = np.array(te['labels']), np.array(te['common_mask'], bool)
    res = {}
    for name, (m, z) in R.items():
        _, model, seed = name.split('__')
        y, p = z['y'].astype(int), z['pred'].astype(int)
        assert np.array_equal(y, labels), 'test order differs from the pipeline test epochs (finetune-sleepedf/results/test_epochs.json)'
        r = {'balanced_acc': balanced_accuracy_score(y, p), 'kappa': cohen_kappa_score(y, p), 'weighted_f1': f1_score(y, p, average='weighted'),
             'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro')}
        r.update({'f1_' + s: f for s, f in zip(STAGES, f1_score(y, p, average=None, labels=range(5)))})
        yc, pc = y[mask], p[mask]
        r.update({'common_balanced_acc': balanced_accuracy_score(yc, pc), 'common_kappa': cohen_kappa_score(yc, pc),
                  'common_weighted_f1': f1_score(yc, pc, average='weighted'), 'common_accuracy': accuracy_score(yc, pc),
                  'common_macro_f1': f1_score(yc, pc, average='macro'), 'common_n_epochs': int(mask.sum())})
        r.update({'common_f1_' + s: f for s, f in zip(STAGES, f1_score(yc, pc, average=None, labels=range(5)))})
        r['val_kappa'] = m['val']['kappa']
        r['minutes'] = m['seconds']['fit_total'] / 60
        res.setdefault(model, {})[int(seed[1:])] = r
        res[model]['_costs'] = costs([m])
    out = {}
    print('Sleep-EDF-78, 62 / 8 / 8 subjects; each 30-s epoch classified on its own (no sequence context). Mean ± SD over seeds.')
    print('pipeline test set (21,080 epochs) | common test set (20,976 epochs)\n')
    for model, per in res.items():
        c = per.pop('_costs')
        keys = per[next(iter(per))].keys()
        out[model] = {'label': LABEL[model], 'implementation': 'ours (XGBoost on dDTF maps)', 'seeds': sorted(per), 'per_seed': per,
                      'mean': {k: ms([per[s][k] for s in per])[0] for k in keys}, 'std': {k: ms([per[s][k] for s in per])[1] for k in keys}, 'costs': c}
        mm, sd = out[model]['mean'], out[model]['std']
        print(f'{LABEL[model]}: ' + '  '.join(f'{k} {mm[k]:.4f} ± {sd[k]:.4f}' for k in ('balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1')))
        print('  common: ' + '  '.join(f'{k} {mm["common_" + k]:.4f} ± {sd["common_" + k]:.4f}' for k in ('balanced_acc', 'kappa', 'macro_f1', 'accuracy', 'weighted_f1')))
        print('  per-stage F1 (pipeline test set): ' + ' '.join(f'{s} {mm["f1_" + s]:.4f}' for s in STAGES))
        print('  ' + fmt_costs(c))
    json.dump({'models': out}, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=float)
    write_csv([row('Sleep-EDF', 'cross-subject 62/8/8', SETTING[k] + ', each 30-s epoch on its own (2 channels)', n, v['mean'][n], v['std'][n],
                   len(v['seeds']), n == 'balanced_acc' and k == 'ddtf-gl') for k, v in out.items() for n in ('balanced_acc', 'kappa', 'weighted_f1')], OUT)


# ------------------------------------------------------------------------------------------------------------------- SEED and SEED-V
def seed(R, OUT):
    trials = json.load(open(os.path.join(CBRAMOD, 'finetune-seed/results/test_trials.json')))
    names = ['balanced_acc', 'kappa', 'weighted_f1', 'accuracy', 'macro_f1', 'clip_vote_acc']
    rows = []
    for name, (m, z) in R.items():
        ds, split, model, s = name.split('__')
        y, p = z['y'].astype(int), z['pred'].astype(int)
        k = {'SEED': 3, 'SEED-V': 5}[ds]
        start, votes = 0, []
        for _, _, n in trials[f'{ds}/{split}']:
            assert len(set(y[start:start + n].tolist())) == 1
            votes.append(np.bincount(p[start:start + n], minlength=k).argmax() == y[start])
            start += n
        assert start == len(y), (start, len(y))
        rows.append(dict(dataset=ds, split=split, model=model, seed=int(s[1:]), metric_names=names,
                         metrics=[balanced_accuracy_score(y, p), cohen_kappa_score(y, p), f1_score(y, p, average='weighted'), accuracy_score(y, p),
                                  f1_score(y, p, average='macro'), float(np.mean(votes))],
                         val_kappa=m['val']['kappa'], per_class_f1=f1_score(y, p, average=None, labels=list(range(k))).tolist(),
                         cm=np.bincount(k * y + p, minlength=k * k).reshape(k, k).tolist(), costs=costs([m]), mvar_order=m['features'].get('order')))
    summary = []
    for key in sorted({(r['dataset'], r['split'], r['model']) for r in rows}):
        rs = sorted((r for r in rows if (r['dataset'], r['split'], r['model']) == key), key=lambda r: r['seed'])
        a = np.array([r['metrics'] for r in rs])
        summary.append(dict(dataset=key[0], split=key[1], model=key[2], n=len(rs), seeds=[r['seed'] for r in rs], metric_names=names,
                            mean=a.mean(0).tolist(), std=(a.std(0, ddof=1) if len(rs) > 1 else np.zeros(len(names))).tolist(), costs=rs[0]['costs']))
        s = summary[-1]
        print(f"{key[0]:7s} {key[1]:9s} {key[2]:17s} n={len(rs)} " + ' '.join(f'{n} {mm:.4f}±{v:.4f}' for n, mm, v in zip(names, s['mean'], s['std'])))
    print()
    for s in summary:
        print(f"{s['dataset']:7s} {s['split']:9s} {fmt_costs(s['costs'])}")
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    PROT = {'trial555': 'all subjects pooled, clips 1-5 / 6-10 / 11-15 of every session', 'subject': None}
    write_csv([row(s['dataset'], PROT[s['split']] or ('cross-subject 9/3/3' if s['dataset'] == 'SEED' else 'cross-subject 10/3/3'),
                   SETTING[s['model']] + ", on the 1-s samples of CBraMod's SEED-V pipeline", n, s['mean'][i], s['std'][i], s['n'],
                   n == 'balanced_acc' and s['model'] == 'ddtf-gl')
               for s in summary for i, n in enumerate(names) if n != 'macro_f1'], OUT)


def main():
    exp, runs_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    R = runs(runs_dir)
    if exp == 'stew-cbramod-pipeline':
        stew(R, out)
    elif exp == 'stew-paper-protocol':
        stew_paper(R, out, runs_dir)
    elif exp == 'cbramod-splits':
        splits(R, out)
    elif exp == 'sleepedf-cbramod-pipeline':
        sleep(R, out)
    elif exp == 'seed-cbramod-pipeline':
        seed(R, out)
    else:
        raise SystemExit(f'unknown experiment {exp}')


if __name__ == '__main__':
    main()
