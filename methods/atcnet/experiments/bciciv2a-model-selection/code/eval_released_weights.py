"""Score the authors' trained ATCNet weights on our copy of BCI IV 2a with the released test() function.

The upstream repository ships one trained model per subject (results/saved models/run-1/subject-<n>.h5, from
main_TrainValTest.py, 1 seed) and its log (results/log.txt: 81.10% / kappa 0.748 on session E). If our data files and
preprocessing match the authors', the same weights give the same predictions. test() of main_TrainValTest.py is
called unchanged; it reads <results_path>/saved models/ and appends its table to <results_path>/log.txt.

usage: python code/eval_released_weights.py [RESULTS_DIR]     # writes RESULTS_DIR/released_weights_log.txt
"""
import os, shutil, sys
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

sys.path.insert(0, paths.UPSTREAM)
import main_TrainValTest

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.HERE, '../results')
tmp = os.path.join(paths.WORK, 'released_weights')
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(os.path.join(paths.UPSTREAM, 'results', 'saved models'), os.path.join(tmp, 'saved models'))
conf = paths.dataset_conf()
main_TrainValTest.test(main_TrainValTest.getModel('ATCNet', conf), conf, tmp)
shutil.copy(os.path.join(tmp, 'log.txt'), os.path.join(RESULTS, 'released_weights_log.txt'))
print('wrote', os.path.join(RESULTS, 'released_weights_log.txt'))
