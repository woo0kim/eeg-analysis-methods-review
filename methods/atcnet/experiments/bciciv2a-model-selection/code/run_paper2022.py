"""Train ATCNet with the code that produced the paper's numbers (upstream commit 3a318b2, 2022-08-18), one subject
and one run, and keep the per-epoch curves.

The body of train() of the 2022 main.py, restricted to one (subject, run); changed lines are marked # [MOD].
getModel() (models.ATCNet with the paper's hyperparameters) and get_data() of that commit are imported unchanged.
Protocol as released then: session T trains, session E is the Keras validation data for ModelCheckpoint
(val_accuracy) and EarlyStopping (val_accuracy, patience 300); Adam lr 0.0009, batch 64, 1000 epochs; no learning-rate
schedule, no weight decay.

Two things of the 2022 loop cannot be kept in a one-run-per-process job:
  * seeds: the 2022 code sets none; here tf/np seed = run number, as the later code does (repeatable runs);
  * one Adam instance is created once in train() and reused by all 90 runs, so every run after the first starts with
    Adam's step counter at the total number of steps before it (>= 1500), where bias correction is (nearly) off.
    Each run here gets a fresh Adam whose counter is set to 100000 (bias-correction factors 1 - 0.9^t and 1 - 0.999^t
    are then 1 to machine precision), except subject 1 / run 1, which in the 2022 loop is the only run that started
    at 0.
usage: python code/run_paper2022.py --subject 1 --seed 1 [--out $ATCNET_WORK/released]
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

sys.path.insert(0, paths.paper_code())
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, cohen_kappa_score
from preprocess import get_data
import main      # the 2022 main.py: getModel()
from curves import Curves

# train_conf of the 2022 run()
TRAIN_CONF = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009, 'n_train': 10, 'model': 'ATCNet'}


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True, help='1..10 = run number of the 2022 loop')
    ap.add_argument('--epochs', type=int, help='override (smoke tests only)')
    ap.add_argument('--out', default=os.path.join(paths.WORK, 'released'))
    a = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 4)))   # [MOD] share CPUs
    tf.config.threading.set_inter_op_parallelism_threads(2)                                        # [MOD]
    if os.environ.get('TF_GPU_MEM_MB'):          # [MOD] cap GPU memory so that several runs can share one GPU
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.set_logical_device_configuration(
                g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ['TF_GPU_MEM_MB']))])
    results_path = os.path.join(a.out, 'paper2022')                                  # [MOD]
    data_path, isStandard, LOSO = paths.data_path(), True, False                     # [MOD] 2022 dataset_conf
    train_conf = dict(TRAIN_CONF, **({'epochs': a.epochs} if a.epochs else {}))

    # ---- from train() of the 2022 main.py ----
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    opt = Adam(learning_rate=lr)
    if (a.subject, a.seed) != (1, 1):            # [MOD] stand-in for the Adam instance shared by all runs (docstring)
        opt.iterations.assign(100000)
    model_name = train_conf.get('model')
    sub, train = a.subject - 1, a.seed - 1                                           # [MOD] one (subject, run)
    print('\nTraining on subject ', sub + 1)
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard)
    tf.random.set_seed(train + 1)                                                    # [MOD] seeds (docstring)
    np.random.seed(train + 1)                                                        # [MOD]
    in_run = time.time()
    filepath = results_path + '/saved models/run-{}'.format(train + 1)
    os.makedirs(filepath, exist_ok=True)
    filepath = filepath + '/subject-{}.h5'.format(sub + 1)
    model = main.getModel(model_name)
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                        save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
    ]
    curves = Curves()                                                                # [MOD]
    model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
              epochs=epochs, batch_size=batch_size, callbacks=callbacks + [curves], verbose=0)   # [MOD] +curves
    model.load_weights(filepath)
    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test_onehot.argmax(axis=-1)
    acc = accuracy_score(labels, y_pred)
    kappa = cohen_kappa_score(labels, y_pred)
    out_run = time.time()
    info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, train + 1, ((out_run - in_run) / 60))
    info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc, kappa)
    print(info)

    # ---- [MOD] keep everything needed to recompute the numbers without a GPU ----
    os.makedirs(os.path.join(results_path, 'curves'), exist_ok=True)
    np.savez_compressed(os.path.join(results_path, 'curves', f's{sub + 1}_seed{train + 1}.npz'),
                        protocol='paper2022', subject=sub + 1, seed=train + 1,
                        **{k: np.array(v, dtype=np.float32) for k, v in curves.h.items()},
                        y_pred=y_pred.astype(np.int8), labels=labels.astype(np.int8), acc=acc, kappa=kappa,
                        minutes=(out_run - in_run) / 60, epochs_run=len(curves.h['loss']),
                        n_train=len(X_train), n_val=len(X_test), n_test=len(X_test), tf_version=tf.__version__)


if __name__ == '__main__':
    main_()
