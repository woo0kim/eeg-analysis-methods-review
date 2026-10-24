"""Step 1 of the equivalence check (TensorFlow env): export the authors' trained Keras ATCNet weights, renamed and
transposed to braindecode's ATCNet state-dict layout, plus the Keras logits on session E, one file per subject.

Two sets of trained weights, each loaded into the model of the code that trained it:
  released  upstream results/saved models/run-1/subject-<n>.h5 (main_TrainValTest.py, today's models.ATCNet_)
  paper     the best-of-10 models behind the paper's 85.38% (upstream commit 3a318b2, 2022 models.ATCNet)
Takes the input of the final softmax as logits and writes $ATCNET_WORK/braindecode/equivalence/<set>_s<n>.npz with
X_test, labels, logits and every weight under its braindecode name. Layers are matched by creation order within each
Keras layer type (both versions build the conv block first, then for each window: LayerNorm, MHA, 4 Conv1D +
4 BatchNorm, Dense).
usage: python code/export_keras.py released; python code/export_keras.py paper      # one process each
       then code/check_equivalence.py in the braindecode env
"""
import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '../../bciciv2a-model-selection/code'))
import paths

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else 'released'
sys.path.insert(0, paths.UPSTREAM if WEIGHTS == 'released' else paths.paper_code())
import tensorflow as tf
from preprocess import get_data
if WEIGHTS == 'released':
    import main_TrainTest
else:
    import main as main_2022

OUT = os.path.join(paths.WORK, 'braindecode', 'equivalence')
N_WINDOWS, TCN_DEPTH = 5, 2


def by_type(model, cls):
    """Layers of one Keras class in creation order (name suffix _k, none = 0)."""
    ls = [l for l in model.layers if type(l).__name__ == cls]
    return sorted(ls, key=lambda l: int(re.search(r'_(\d+)$', l.name).group(1)) if re.search(r'_\d+$', l.name) else 0)


def bn(prefix, layer):
    g, b, m, v = layer.get_weights()
    return {f'{prefix}.weight': g, f'{prefix}.bias': b, f'{prefix}.running_mean': m, f'{prefix}.running_var': v}


def to_braindecode(model):
    sd = {}
    conv2d, dw, bns = by_type(model, 'Conv2D'), by_type(model, 'DepthwiseConv2D'), by_type(model, 'BatchNormalization')
    lns, mhas, conv1d, dense = (by_type(model, c) for c in ('LayerNormalization', 'MultiHeadAttention', 'Conv1D', 'Dense'))
    assert (len(conv2d), len(dw), len(bns), len(lns), len(mhas), len(conv1d), len(dense)) == \
        (2, 1, 3 + 2 * TCN_DEPTH * N_WINDOWS, N_WINDOWS, N_WINDOWS, 2 * TCN_DEPTH * N_WINDOWS, N_WINDOWS)
    # conv block: Keras kernels are (kh, kw, in, out) on (T, C, 1) inputs; torch (out, in, kh, kw) on (1, T, C)
    sd['conv_block.conv1.weight'] = conv2d[0].get_weights()[0].transpose(3, 2, 0, 1)
    k = dw[0].get_weights()[0]                                   # (1, C, F1, D): output channel f*D + m
    sd['conv_block.conv2.weight'] = k.transpose(2, 3, 0, 1).reshape(-1, 1, 1, k.shape[1])
    sd['conv_block.conv3.weight'] = conv2d[1].get_weights()[0].transpose(3, 2, 0, 1)
    for i in range(3):
        sd.update(bn(f'conv_block.bn{i + 1}', bns[i]))
    for w in range(N_WINDOWS):
        g, b = lns[w].get_weights()
        sd[f'attention_blocks.{w}.ln.weight'], sd[f'attention_blocks.{w}.ln.bias'] = g, b
        mw = {v.name.split('/')[-2] + '/' + v.name.split('/')[-1].split(':')[0]: v.numpy() for v in mhas[w].weights}
        for kname, t in (('query', 'q'), ('key', 'k'), ('value', 'v')):       # (in, heads, dh) -> (heads*dh, in)
            kk = mw[f'{kname}/kernel']
            sd[f'attention_blocks.{w}.mha.fc_{t}.weight'] = kk.reshape(kk.shape[0], -1).T
            sd[f'attention_blocks.{w}.mha.fc_{t}.bias'] = mw[f'{kname}/bias'].reshape(-1)
        ko = mw['attention_output/kernel']                                    # (heads, dh, out) -> (out, heads*dh)
        sd[f'attention_blocks.{w}.mha.fc_o.weight'] = ko.reshape(-1, ko.shape[-1]).T
        sd[f'attention_blocks.{w}.mha.fc_o.bias'] = mw['attention_output/bias']
        for i in range(TCN_DEPTH):
            for j in (1, 2):
                n = w * 2 * TCN_DEPTH + 2 * i + (j - 1)
                kk, bb = conv1d[n].get_weights()                              # (k, in, out) -> (out, in, k)
                sd[f'temporal_conv_nets.{w}.{i}.conv{j}.weight'] = kk.transpose(2, 1, 0)
                sd[f'temporal_conv_nets.{w}.{i}.conv{j}.bias'] = bb
                sd.update(bn(f'temporal_conv_nets.{w}.{i}.bn{j}', bns[3 + n]))
        kk, bb = dense[w].get_weights()                                       # (in, out) -> (out, in)
        sd[f'final_layer.{w}.weight'], sd[f'final_layer.{w}.bias'] = kk.T, bb
    return sd


def main():
    os.makedirs(OUT, exist_ok=True)
    conf = paths.dataset_conf()
    if WEIGHTS == 'paper':
        best = [l.strip() for l in open(os.path.join(paths.paper_code(), 'results', 'best models.txt')) if l.strip()]
    for s in range(1, 10):
        tf.keras.backend.clear_session()        # restart layer-name counters
        if WEIGHTS == 'released':
            model = main_TrainTest.getModel('ATCNet', conf)
            model.load_weights(os.path.join(paths.UPSTREAM, 'results', 'saved models', 'run-1', f'subject-{s}.h5'))
            _, _, _, X_test, y_test, _ = get_data(conf['data_path'], s - 1, 'BCI2a', LOSO=False, isStandard=True)
        else:
            model = main_2022.getModel('ATCNet')
            model.load_weights(os.path.join(paths.paper_code(), 'results') + best[s - 1])
            _, _, _, X_test, y_test, _ = get_data(conf['data_path'], s - 1, False, True)
            y_test = (y_test - 1).astype(int)        # the 2022 get_data returns labels 1..4
        logit_model = tf.keras.Model(model.input, model.get_layer('softmax').input)
        logits = logit_model.predict(X_test, batch_size=288)
        probs = model.predict(X_test, batch_size=288)
        sd = to_braindecode(model)
        n_trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
        np.savez(os.path.join(OUT, f'{WEIGHTS}_s{s}.npz'), X_test=X_test.astype(np.float32), labels=y_test,
                 logits=logits, probs=probs, keras_params=model.count_params(), keras_trainable=n_trainable,
                 **{'sd/' + k: v for k, v in sd.items()})
        print(f'{WEIGHTS} subject {s}: Keras acc {(probs.argmax(1) == y_test).mean():.4f}, {len(sd)} tensors, '
              f'{model.count_params()} params ({n_trainable} trainable)')


if __name__ == '__main__':
    main()
