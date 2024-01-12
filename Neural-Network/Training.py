import os
import time
import tensorflow as tf
from tensorflow.keras import losses

from Constants import Dataset, Hyperparameters
from Dataset import create_tf_dataset
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
    # TODO: add learning rate scheduler

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=Hyperparameters.learning_rate)

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
    dataset = Dataset.FILENAME.split(".")[0]
    folder_name = f"Model-{Hyperparameters.learning_rate}-LR-{dataset}-DS-" + current_time
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


def start_training():
    dataset = create_tf_dataset(Dataset.PATH)
    model = get_model()
    model = compile_model(model)
    callbacks, checkpoint_dir = create_callbacks()


    # Training
    model.fit(x=dataset,
              epochs=Hyperparameters.EPOCHS,
              initial_epoch=0,
              shuffle=True,
              verbose=1,
              callbacks=callbacks)


if __name__ == "__main__":
    # create_tensorflow_session(0.5)
    start_training()
