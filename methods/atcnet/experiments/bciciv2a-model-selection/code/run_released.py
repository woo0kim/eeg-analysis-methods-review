"""Train the released ATCNet (TensorFlow) for one subject and one seed under one of the released protocols.

  traintest     main_TrainTest.py as released, the protocol of the paper: session T trains, and session E is passed to
                model.fit as validation data, so the checkpoint (best val_accuracy), ReduceLROnPlateau (val_loss) and
                EarlyStopping (val_accuracy, patience 300) all look at the test set. 1000 epochs.
  trainvaltest  main_TrainValTest.py as released, the protocol the README now recommends: 20% of session T
                (train_test_split, random_state=42) is the validation set, checkpoint on val_loss, no early stopping,
                session E only for the final test. 500 epochs (its run()).
  loso          main_TrainTest.py with LOSO=True (paper, Table VI): the other 8 subjects (both sessions) train, the
                held-out subject (both sessions) is validation and test set.

The body of train() is copied from the upstream file of the protocol for a single (subject, seed); every changed line
is marked # [MOD]. getModel() (models.ATCNet_ with the released hyperparameters) and get_data() are the upstream
functions, imported unchanged. Seeds follow the released loop: run `train` uses tf/np seed train+1, so --seed 1..10 is
runs 1..10.

Writes <out>/<protocol>/saved models/run-<seed>/subject-<sub>.h5 (the layout the released test() reads) and
<out>/<protocol>/curves/s<sub>_seed<seed>.npz (per-epoch curves, test predictions).
usage: python code/run_released.py --protocol traintest --subject 1 --seed 1 [--out $ATCNET_WORK/released]
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

sys.path.insert(0, paths.UPSTREAM)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from preprocess import get_data
import main_TrainTest
from curves import Curves

# train_conf of the released run() of each file (dataset 'BCI2a'); LearnCurves only plots, so it is off here
TRAIN_CONF = {
    'traintest': {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.001, 'n_train': 10, 'model': 'ATCNet'},
    'trainvaltest': {'batch_size': 64, 'epochs': 500, 'patience': 100, 'lr': 0.001, 'n_train': 1,
                     'from_logits': False, 'model': 'ATCNet'},
}
TRAIN_CONF['loso'] = TRAIN_CONF['traintest']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=sorted(TRAIN_CONF), required=True)
    ap.add_argument('--subject', type=int, required=True, help='1..9')
    ap.add_argument('--seed', type=int, required=True, help='1..10 = run number of the released loop')
    ap.add_argument('--epochs', type=int, help='override (smoke tests only)')
    ap.add_argument('--out', default=os.path.join(paths.WORK, 'released'))
    a = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('TF_THREADS', 4)))   # [MOD] share CPUs
    tf.config.threading.set_inter_op_parallelism_threads(2)                                        # [MOD]
    if os.environ.get('TF_GPU_MEM_MB'):          # [MOD] cap GPU memory so that several runs can share one GPU
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.set_logical_device_configuration(
                g, [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ['TF_GPU_MEM_MB']))])

    dataset_conf = paths.dataset_conf(loso=a.protocol == 'loso')                     # [MOD] released values, our paths
    train_conf = dict(TRAIN_CONF[a.protocol], **({'epochs': a.epochs} if a.epochs else {}))
    results_path = os.path.join(a.out, a.protocol)                                   # [MOD]

    # ---- from train() of main_TrainTest.py / main_TrainValTest.py ----
    dataset = dataset_conf.get('name')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    model_name = train_conf.get('model')

    sub, train = a.subject - 1, a.seed - 1                                           # [MOD] one (subject, run)
    print('\nTraining on subject ', sub + 1)
    X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
        data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
    if a.protocol == 'trainvaltest':
        # main_TrainValTest.py: divide the training data into training and validation
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2,
                                                                        random_state=42)
    else:
        X_val, y_val_onehot = X_test, y_test_onehot   # main_TrainTest.py: validation_data=(X_test, y_test_onehot)

    tf.random.set_seed(train + 1)
    np.random.seed(train + 1)
    in_run = time.time()
    filepath = results_path + '/saved models/run-{}'.format(train + 1)
    os.makedirs(filepath, exist_ok=True)
    filepath = filepath + '/subject-{}.h5'.format(sub + 1)

    model = main_TrainTest.getModel(model_name, dataset_conf)   # identical in both files for from_logits=False
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    if a.protocol == 'trainvaltest':
        callbacks = [
            ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=True, mode='min'),
            ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
            # EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=patience)
        ]
    else:
        callbacks = [
            ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                            save_best_only=True, save_weights_only=True, mode='max'),
            ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=1, min_lr=0.0001),
            EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
        ]
    labels_test = y_test_onehot.argmax(axis=-1)
    curves = Curves(X_test, labels_test) if a.protocol == 'trainvaltest' else Curves()   # [MOD]
    history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks + [curves], verbose=0)  # [MOD] +curves

    model.load_weights(filepath)
    y_pred = model.predict(X_test).argmax(axis=-1)       # what test() of either file computes from the saved model
    acc = accuracy_score(labels_test, y_pred)
    kappa = cohen_kappa_score(labels_test, y_pred)
    out_run = time.time()
    info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, train + 1, ((out_run - in_run) / 60))
    info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc, kappa)
    extra = {}
    if a.protocol == 'trainvaltest':                     # main_TrainValTest.py prints the validation accuracy
        y_val_pred = model.predict(X_val).argmax(axis=-1)
        extra['val_acc_at_checkpoint'] = accuracy_score(y_val_onehot.argmax(axis=-1), y_val_pred)
        info = info + '   valid_acc: {:.4f}   valid_loss: {:.3f}'.format(extra['val_acc_at_checkpoint'],
                                                                         min(history.history['val_loss']))
    print(info)

    # ---- [MOD] keep everything needed to recompute the numbers without a GPU ----
    os.makedirs(os.path.join(results_path, 'curves'), exist_ok=True)
    np.savez_compressed(os.path.join(results_path, 'curves', f's{sub + 1}_seed{train + 1}.npz'),
                        protocol=a.protocol, subject=sub + 1, seed=train + 1,
                        **{k: np.array(v, dtype=np.float32) for k, v in curves.h.items()},
                        y_pred=y_pred.astype(np.int8), labels=labels_test.astype(np.int8), acc=acc, kappa=kappa,
                        minutes=(out_run - in_run) / 60, epochs_run=len(curves.h['loss']),
                        n_train=len(X_train), n_val=len(X_val), n_test=len(X_test),
                        tf_version=tf.__version__, **extra)


if __name__ == '__main__':
    main()
