import os
import time
import tensorflow as tf
from tensorflow.keras import losses

from Constants import Dataset, Hyperparameters
from Dataset import create_tf_dataset, create_dataset
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


def train_step(model, optimizer, batch):
    losses = []
    batch_inputs, batch_outputs = batch
    batch_forces, batch_total_energy = batch_outputs

    # loop over the elements of the batch
    for i in range(Dataset.BATCH_SIZE):
        inputs = [batch_inputs[k][i] for k in range(4)]
        forces = batch_forces[i]
        total_energy = batch_total_energy[i]

        total_energy_pred = 0

        num_atoms = inputs[0].shape[1]

        force_losses = []

        # Loop over atoms
        for j in range(num_atoms):
            input_atom = [inputs[k]for k in range(4)]
            force = forces[j]

            # Calculate predictions
            model_output = model(input_atom)
            force_pred, energy_pred = model_output[:, :3], model_output[:, 3]

            # Calculate losses
            force_loss = tf.keras.losses.MSE(force, force_pred)

            force_losses.append(force_loss)
            total_energy_pred += energy_pred

        # Calculate total loss
        loss = tf.math.reduce_sum(force_losses) + tf.keras.losses.MSE(total_energy, total_energy_pred)
        losses.append(loss)

    # Compute and apply gradients
    with tf.GradientTape() as tape:
        gradients = tape.gradient(losses, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Training function
def start_training_custom():
    dataset = create_dataset(Dataset.PATH)
    model = get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(Hyperparameters.EPOCHS):
        for batch in dataset:
            train_step(model, optimizer, batch)

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
    # start_training()
    start_training_custom()
