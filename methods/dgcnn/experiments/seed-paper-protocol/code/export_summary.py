"""Write results/summary.csv from results/results.json in the repo-wide schema (docs/adding-an-experiment.md).
usage: python code/export_summary.py [results]"""
import csv, json, os, statistics, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else 'results'
R = json.load(open(os.path.join(RESULTS, 'results.json')))
SD, LOSO = 'within-subject, session by session, first 9 clips -> last 6', 'leave one subject out, one session at a time'
rows = []


def row(protocol, setting, mean, std, n, paper=None, head=0):
    rows.append(dict(method='DGCNN', dataset='SEED', protocol=protocol, setting=setting, metric='accuracy',
                     paper_mean=paper[0] if paper else '', paper_std=paper[1] if paper else '', paper_source='abstract' if paper else '',
                     ours_mean=round(mean / 100, 4), ours_std=round(std / 100, 4) if std is not None else '', n_runs=n, headline=head))


for r in R:
    if r.get('feature') == 'de_LDS' and r.get('rule') == 'peak on test':
        v, a = r['best2_per_run'], r['all3_per_run']
        row(SD, "our PyTorch port, authors' settings and rule: peak of ~80 test evaluations, best 2 of 3 sessions", r['best2_mean'],
            statistics.stdev(v), r['n_runs'], (0.9040, 0.0849), 1)
        row(SD, "our PyTorch port, authors' settings: peak of ~80 test evaluations, all 3 sessions", r['all3_mean'], statistics.stdev(a), r['n_runs'])
    if r.get('feature') == 'de_LDS' and r.get('rule') == 'last step':
        row(SD, "our PyTorch port, authors' settings (lr 0.07): last step, all 3 sessions", r['all3_mean'], statistics.stdev(r['all3_per_run']), r['n_runs'])
    if r.get('feature') == 'de_LDS' and r.get('rule') == 'best validation step':
        row(SD, "our PyTorch port, authors' settings: step chosen on clips 8-9, all 3 sessions", r['all3_mean'], statistics.stdev(r['all3_per_run']), r['n_runs'])
    if 'picked' in r:
        p = r['picked']
        row(SD, f"our PyTorch port, lr {p['lr']} chosen on clips 8-9: last step, all 3 sessions", p['test_last'], None, 5, None, 1)
    if r.get('protocol') == 'loso-sessions' and r.get('feature') == 'de_LDS' and r.get('norm') == 'zscore':
        row(LOSO, "our PyTorch port, authors' lr 0.07, sessions z-scored: peak over test evaluations", r['peak_mean'], None, r['n_runs'], (0.7995, 0.0902))
    if r.get('protocol') == 'loso-sessions' and r.get('feature') == 'de_LDS' and r.get('norm') == 'zscore-lr0.01':
        row(LOSO, 'our PyTorch port, lr 0.01, sessions z-scored: last step', r['last_mean'], None, r['n_runs'], (0.7995, 0.0902), 1)
with open(os.path.join(RESULTS, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows))
