import os
import time
import tensorflow as tf
from tensorflow.keras import losses
from tqdm import tqdm

from Constants import Dataset, Hyperparameters
from Dataset_NP import create_tf_generator_dataset, create_tf_dataset
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


def train_step(model, optimizer, element):
    inputs, outputs = element
    forces, total_energy = outputs

    # Convert forces to float32 to match the model output
    forces = tf.cast(forces, dtype=tf.float32)

    # Watch the model's trainable variables
    with tf.GradientTape() as tape:
        # Make predictions
        prediction = model([inputs[0], inputs[1], inputs[2], inputs[3]])

        # Separate energy and force predictions
        energy_pred = prediction[:, 0]
        force_pred = prediction[:, 1:]

        # Total energy is the sum of atomic energies
        total_energy_pred = tf.math.reduce_sum(energy_pred)

        # Calculate losses
        total_energy_loss = losses.MSE(total_energy, total_energy_pred)
        force_loss = losses.MSE(forces, force_pred)

        # Total loss is the sum of total energy loss and force loss
        total_loss = total_energy_loss + force_loss

        # Print mean loss
        print(f"Total loss: {tf.reduce_mean(total_loss)}")

    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Training function
def start_training_custom():
    dataset = create_tf_generator_dataset(Dataset.PATH)
    model = get_model(predict_force_only=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    dataset_size = None
    for epoch in range(Hyperparameters.EPOCHS):
        i = 0
        for element in dataset:
            print(f"Epoch {epoch} - Training step {i}/{dataset_size if dataset_size else 'Unknown'}")
            train_step(model, optimizer, element)
            i += 1
        dataset_size = i

        # Save the model after each epoch
        model.save(os.path.join("..", "Checkpoints", f"model_epoch_{epoch}.h5"))
        print(f"Epoch {epoch} finished")


def compile_model(model):
    # TODO: add learning rate scheduler
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


def start_training():
    train_ds = create_tf_dataset(os.path.join(Dataset.FOLDER, Dataset.TRAIN_NAME))
    val_ds = create_tf_dataset(os.path.join(Dataset.FOLDER, Dataset.VAL_NAME))
    model = get_model(predict_force_only=True)
    model = compile_model(model)
    callbacks, checkpoint_dir = create_callbacks()


    # Training
    model.fit(x=train_ds,
              validation_data=val_ds,
              epochs=Hyperparameters.EPOCHS,
              initial_epoch=0,
              shuffle=True,
              verbose=1,
              callbacks=callbacks)


if __name__ == "__main__":
    # create_tensorflow_session(0.5)
    start_training()
    # start_training_custom()
