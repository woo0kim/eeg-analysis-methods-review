"""Paths shared by the scripts of this experiment (defaults = the mcl-server layout of the committed runs).

  EEG_DATA_ROOT  holds bciciv2a/data_mat/A0{1..9}{T,E}.mat (datasets/download.sh bciciv2a)     default ~/data
  ATCNET_WORK    saved models, per-run outputs and the data layout the released code expects  default ~/repro-atcnet
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.normpath(os.path.join(HERE, '../../../upstream'))
EEG_DATA_ROOT = os.path.expanduser(os.environ.get('EEG_DATA_ROOT', '~/data'))
WORK = os.path.expanduser(os.environ.get('ATCNET_WORK', '~/repro-atcnet'))
CLASSES = ['Left hand', 'Right hand', 'Foot', 'Tongue']


def data_path():
    """The released preprocess.py reads <data_path>/s<N>/A0<N>{T,E}.mat (the BNCI 001-2014 files, one folder per
    subject). Build that layout once as symlinks to $EEG_DATA_ROOT/bciciv2a/data_mat and return <data_path>."""
    root = os.path.join(WORK, 'data')
    for s in range(1, 10):
        os.makedirs(f'{root}/s{s}', exist_ok=True)
        for t in 'TE':
            src, dst = f'{EEG_DATA_ROOT}/bciciv2a/data_mat/A0{s}{t}.mat', f'{root}/s{s}/A0{s}{t}.mat'
            if not os.path.exists(src):
                raise FileNotFoundError(f'{src} (run datasets/download.sh bciciv2a)')
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
    return root + '/'


def dataset_conf(loso=False):
    """dataset_conf of the released run() for dataset = 'BCI2a' (main_TrainTest.py and main_TrainValTest.py agree)."""
    return {'name': 'BCI2a', 'n_classes': 4, 'cl_labels': CLASSES, 'n_sub': 9, 'n_channels': 22, 'in_samples': 1125,
            'data_path': data_path(), 'isStandard': True, 'LOSO': loso}


# The code that produced the paper's numbers: upstream commit 3a318b2 (2022-08-18) uploaded main.py, models.py,
# attention_models.py, preprocess.py and results/ (log, best-of-10 models); the results were deleted on 2023-12-31
# and the training code was rewritten on 2023-12-28 (a4a68f9). It is read from the submodule's history.
PAPER_COMMIT = '3a318b2'


def paper_code():
    """Extract the 2022 code and results of PAPER_COMMIT to $ATCNET_WORK/upstream-2022 (once) and return the path."""
    import subprocess, tempfile
    dst = os.path.join(WORK, 'upstream-2022')
    if not os.path.isdir(dst):
        os.makedirs(WORK, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=WORK)
        tar = subprocess.run(['git', '-C', UPSTREAM, 'archive', PAPER_COMMIT, 'main.py', 'models.py',
                              'attention_models.py', 'preprocess.py', 'results'], check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', tmp], input=tar, check=True)
        try:
            os.rename(tmp, dst)
        except OSError:          # another job extracted it first
            subprocess.run(['rm', '-rf', tmp])
    return dst
