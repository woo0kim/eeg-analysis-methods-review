"""Score the models behind the paper's 85.38% on our copy of BCI IV 2a, with the 2022 test() function.

Upstream commit 3a318b2 (2022-08-18, nine days after the paper went online) holds results/log.txt of the 10 runs per
subject that end in "Average all subjects - best run: Accuracy = 0.8538 Kappa = 0.8050", and the best run's weights
per subject (results/saved models/run-<k>/subject-<n>.h5, listed in results/best models.txt). test() of that
commit's main.py is called unchanged on a copy of its results folder.
usage: python code/eval_paper_weights.py [RESULTS_DIR]       # writes RESULTS_DIR/paper_weights_log.txt
"""
import os, shutil, sys
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

code2022 = paths.paper_code()
sys.path.insert(0, code2022)
import main     # the 2022 main.py

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.HERE, '../results')
tmp = os.path.join(paths.WORK, 'paper_weights')
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(os.path.join(code2022, 'results'), tmp)
open(os.path.join(tmp, 'log.txt'), 'w').close()           # test() appends; keep only today's output
conf = {'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': paths.data_path(), 'isStandard': True, 'LOSO': False}
main.test(main.getModel('ATCNet'), conf, tmp)
shutil.copy(os.path.join(tmp, 'log.txt'), os.path.join(RESULTS, 'paper_weights_log.txt'))
print('wrote', os.path.join(RESULTS, 'paper_weights_log.txt'))
