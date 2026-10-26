"""Keras callback that records every epoch of a run (the released scripts keep only the history plot)."""
import numpy as np
import tensorflow as tf


class Curves(tf.keras.callbacks.Callback):
    def __init__(self, X_test=None, y_test=None):
        super().__init__()
        self.X_test, self.y_test = X_test, y_test
        self.h = {k: [] for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr', 'test_acc')}

    def on_epoch_begin(self, epoch, logs=None):
        self.h['lr'].append(float(tf.keras.backend.get_value(self.model.optimizer.lr)))   # lr used in this epoch

    def on_epoch_end(self, epoch, logs=None):
        for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
            self.h[k].append(float(logs[k]))
        if self.X_test is not None:   # *trainvaltest: test accuracy per epoch, for analysis, never for selection
            # model() in batches, not model.predict(): in TF 2.7 every predict() call keeps ~8 MB of GPU memory,
            # which ran the LOSO runs (576 test trials) out of their 5 GB GPU share late in training
            p = np.concatenate([self.model(self.X_test[i:i + 96], training=False).numpy()
                                for i in range(0, len(self.X_test), 96)]).argmax(axis=-1)
            self.h['test_acc'].append(float((p == self.y_test).mean()))
