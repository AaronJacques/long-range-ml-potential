import os
import time

import tensorflow as tf
from tensorflow.keras import losses

from Constants import Dataset, Hyperparameters
from Dataset import create_tf_dataset_force_only
from Model import get_model


# loss function
class ForceMSE(losses.Loss):
    def __init__(
            self,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='force_mse',
            **kwargs
    ):
        super(ForceMSE, self).__init__(reduction=reduction, name=name)
        self.batch_size = Dataset.BATCH_SIZE

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            'batch_size': self.batch_size,
        }
        base_config = super(ForceMSE, self).get_config()
        return dict(
            list(base_config.items()) + list(config.items())
        )

    @tf.function
    def __call__(
            self,
            y_true,
            y_pred,
            sample_weight=None
    ):
        # y_true is the true force
        # y_pred is the predicted force
        # y_true and y_pred are of shape (batch_size, 3)

        # calculate the mean squared error
        mse = tf.keras.losses.MSE(y_true, y_pred)

        return mse


def compile_model(model):
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=Hyperparameters.learning_rate,
        decay_steps=Hyperparameters.decay_steps,
        decay_rate=Hyperparameters.decay_rate,
        staircase=True
    )

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    # loss for each output
    loss_func = ForceMSE()

    # Compiling the model
    model.compile(optimizer=optimizer,
                  loss=loss_func)

    model.summary()

    return model


def create_callbacks():
    current_time = str(
        # time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime(time.time()))
        time.strftime("%H-%M-%S", time.localtime(time.time()))
    )
    folder_name = f"Model-{Hyperparameters.learning_rate}-LR-{Dataset.NAME}-DS-" + current_time
    checkpoint_path = os.path.join("..", "Checkpoints", folder_name, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print(checkpoint_dir)
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_pck',
        save_best_only=False,
        save_weights_only=False,
        verbose=1)
    cpB = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(checkpoint_dir, "logs")
    )

    return [cp, cpB], checkpoint_dir


def start_training(checkpoint_path=None, initial_epoch=0):
    train_ds = create_tf_dataset_force_only(os.path.join(Dataset.FOLDER, Dataset.TRAIN_NAME))
    val_ds = create_tf_dataset_force_only(os.path.join(Dataset.FOLDER, Dataset.VAL_NAME))
    callbacks, checkpoint_dir = create_callbacks()

    if checkpoint_path:
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={"ForceMSE": ForceMSE}
        )
    else:
        model = get_model(predict_force_only=True)
        model = compile_model(model)


    # Training
    model.fit(x=train_ds,
              validation_data=val_ds,
              epochs=Hyperparameters.EPOCHS,
              initial_epoch=initial_epoch,
              shuffle=True,
              verbose=1,
              callbacks=callbacks)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    start_training(
        #checkpoint_path=r"G:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Model-0.0001-LR-df-DS-17-55-40\cp-0118.ckpt",
        #initial_epoch=118
    )
