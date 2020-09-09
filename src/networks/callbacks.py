import os
from datetime import date
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../eval'))
from pascal import Evaluate


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_callbacks(prediction_model, validation_generator):
    today = str(date.today())
    callbacks = []
    tensorboard_callback = TensorBoard(log_dir='logs/{}'.format(today))
    callbacks.append(tensorboard_callback)

    evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback, score_threshold=0.5)
    callbacks.append(evaluation)

    makedirs('checkpoints/{}'.format(today))
    checkpoint = ModelCheckpoint(os.path.join('checkpoints/{}'.format(today), 'ep{epoch:02d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5'), verbose=1, save_weights_only=True, save_best_only=False)
    callbacks.append(checkpoint)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    callbacks.append(reduce_lr)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    callbacks.append(early_stopping)

    return callbacks
