"""Run a released L-SeqSleepNet script with its evaluation batch size (and, optionally, evaluation interval) changed.

usage (from the released network/lseqsleepnet directory, like the released run scripts):
  python run_released.py train_lseqsleepnet.py|test_lseqsleepnet.py --mod_eval_factor=<F> [--mod_evaluate_every=<N>]
                         <released flags>

[MOD] eval factor: the released train/test scripts evaluate in batches of `factor * batch_size` sequences with
factor = 20*4, i.e. 640 sequences x 200 epochs = 128,000 epochs per forward pass; on a 12 GB Quadro M6000 at the
released --gpu_usage 0.45 (two folds per GPU) the first validation pass runs out of memory. The released line is
replaced by `factor = <F>`. Evaluation runs with dropout off and batch normalisation in inference mode
(istraining 0), so the batch size changes throughput, not the predictions.

[MOD] evaluate_every (only if given): config.py validates every 100 training steps. The released Sleep-EDF-20 scripts
stop after 110 validations (--max_eval_steps 110, 11,000 steps). On the larger Sleep-EDF-78 training set, scaling the
training length by the data size (44,000 steps) at 100 steps per validation would mean 440 full validation passes
(each scores every 200-epoch window of the validation nights, ~4 min on an M6000); --mod_evaluate_every=400 keeps the
released 110 validations over the longer run. The released script stops at max_eval_steps * evaluate_every steps.
Training, checkpoint selection (best validation accuracy) and testing are otherwise the released code.
"""
import os
import sys

PATCHES = {'--mod_eval_factor': ('factor = 20*4', 'factor = {}  # [MOD] released: 20*4'),
           '--mod_evaluate_every': ('config.max_eval_steps = FLAGS.max_eval_steps',
                                    'config.max_eval_steps = FLAGS.max_eval_steps\n'
                                    'config.evaluate_every = {}  # [MOD] released: 100 (config.py)')}

script = os.path.abspath(sys.argv[1])
src, rest = open(script).read(), []
for a in sys.argv[2:]:
    flag, _, value = a.partition('=')
    if flag in PATCHES:
        old, new = PATCHES[flag]
        assert src.count(old) == 1, f'{script}: expected one "{old}"'
        src = src.replace(old, new.format(int(value)))
    else:
        rest.append(a)
assert 'factor = 20*4' not in src, 'pass --mod_eval_factor (the released evaluation batch does not fit in 12 GB)'
sys.argv = [script] + rest
sys.path.insert(0, os.path.dirname(script))
exec(compile(src, script, 'exec'), {'__name__': '__main__', '__file__': script})
