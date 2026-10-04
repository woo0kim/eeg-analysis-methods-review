"""Create runnable copies of the released preprocessing scripts.

Only two kinds of edits are made (each asserted to match exactly once):
  1. hard-coded author paths (/data/...) -> this server's paths
  2. LMDB map_size -> 20 GB (capacity only; LMDB files are sparse, data are unchanged).
     The released SHU-MI script reserves 110,612,736 B (~105 MB) for ~2.5 GB of samples,
     which raises lmdb.MapFullError.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.environ.get('CBRAMOD_SRC', os.path.join(HERE, '../../../upstream')), 'preprocessing')
DST = os.path.join(os.path.expanduser(os.environ.get('CBRAMOD_WORK', '~/repro')), 'preprocessing')
D = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
os.makedirs(DST, exist_ok=True)

EDITS = {
    'preprocessing_physio.py': [
        ("'/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/files'", f"'{D}/physio/files'"),
        ("'/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average'", f"'{D}/physio/processed_average'"),
    ],
    'preprocessing_shu.py': [
        ("'/data/datasets/shu_datasets/mat'", f"'{D}/shu/mat'"),
        ("'/data/datasets/shu_datasets/processed'", f"'{D}/shu/processed'"),
    ],
    'preprocessing_stress.py': [
        ("'/data/datasets/BigDownstream/mental-arithmetic/edf'", f"'{D}/mental_arithmetic/edf'"),
        ("'/data/datasets/BigDownstream/mental-arithmetic/processed'", f"'{D}/mental_arithmetic/processed'"),
    ],
    'preprocessing_mumtaz.py': [
        ("'/data/datasets/MDDPHCED/files'", f"'{D}/mumtaz/files'"),
        ("'/data/datasets/MDDPHCED/processed_lmdb_75hz'", f"'{D}/mumtaz/processed_lmdb_75hz'"),
    ],
    'preprocessing_bciciv2a.py': [
        ("'/data/wjq/datasets/BigDownstream/BCICIV2a/data_mat'", f"'{D}/bciciv2a/data_mat'"),
        ("'/data/wjq/datasets/BigDownstream/BCICIV2a/processed_inde_avg_filter'", f"'{D}/bciciv2a/processed_inde_avg_filter'"),
    ],
}

for name, edits in EDITS.items():
    code = open(os.path.join(SRC, name)).read()
    for old, new in edits:
        assert code.count(old) == 1, (name, old, code.count(old))
        code = code.replace(old, new)
    code, n = re.subn(r'map_size=\d+', 'map_size=20 * 1024**3', code)
    assert n == 1, (name, n)
    open(os.path.join(DST, name), 'w').write(code)
    print('wrote', os.path.join(DST, name))
