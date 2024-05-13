import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tqdm import tqdm

from Constants import Dataset, Hyperparameters, Logging, Model
from Dataset import create_tf_dataset, create_tf_dataset_force_only
from Model import get_model


def get_weighting_from_step(step, w_limit, w_start, decay_steps, decay_rate):
    f = decay_rate**(step / decay_steps)
    return w_limit * (1 - f) + w_start * f


def create_training_directory():
    current_time = str(
        # time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime(time.time()))
        time.strftime("%H-%M-%S", time.localtime(time.time()))
    )
    folder_name = f"Model-{Dataset.FOLDER_NAME}-DS-{Hyperparameters.initial_learning_rate}-lr-{Hyperparameters.decay_steps}-decay-steps" + current_time
    checkpoint_dir = os.path.join("..", "Checkpoints", folder_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    return checkpoint_dir


# logging function
def log_training_progress(
        epoch, train_loss, train_energy_loss, train_force_loss,
        p_energy, p_force,
        val_loss=None, val_energy_loss=None, val_force_loss=None,
        directory="logs"
):
    print("Saving training progress")
    # write the training progress to the log file
    with open(os.path.join(directory, "training_log.csv"), "a") as file:
        # if the file is empty, write the header
        if val_loss is not None:
            if file.tell() == 0:
                file.write(f"{Logging.epoch_key},{Logging.train_total_loss_key},{Logging.train_energy_loss_key},{Logging.train_force_loss_key},{Logging.val_total_loss_key},{Logging.val_energy_loss_key},{Logging.val_force_loss_key},{Logging.p_energy_key},{Logging.p_force_key}\n")
            file.write(f"{epoch},{train_loss},{train_energy_loss},{train_force_loss},{val_loss},{val_energy_loss},{val_force_loss},{p_energy},{p_force}\n")
        else:
            if file.tell() == 0:
                file.write(f"{Logging.epoch_key},{Logging.train_total_loss_key},{Logging.train_energy_loss_key},{Logging.train_force_loss_key},{Logging.p_energy_key},{Logging.p_force_key}\n")
            file.write(f"{epoch},{train_loss},{train_energy_loss},{train_force_loss},{p_energy},{p_force}\n")

    print("Training progress saved")


# Training function
def start_training():
    @tf.function
    def train_step(inputs, outputs, p_energy, p_force):
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

            # Calculate energy loss
            total_energy_loss = tf.reduce_mean(tf.square(total_energy_pred - total_energy)) # losses.MSE(total_energy, total_energy_pred)

            # Calculate force loss
            force_pred_reshape = tf.reshape(force_pred, [-1])
            forces_reshape = tf.reshape(forces, [-1])
            diff_f = forces_reshape - force_pred_reshape
            force_loss = tf.reduce_mean(tf.square(diff_f))

            # Total loss is the sum of total energy loss and force loss
            total_loss = p_energy * total_energy_loss + p_force * force_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss, total_energy_loss, force_loss

    @tf.function
    def train_step_energy_only(inputs, outputs):
        # Watch the model's trainable variables
        with tf.GradientTape() as tape:
            # Make predictions
            prediction = model([inputs[0], inputs[1], inputs[2], inputs[3]])

            # Total energy is the sum of atomic energies
            total_energy_pred = tf.math.reduce_sum(prediction)

            # Calculate energy loss
            total_energy_loss = tf.reduce_mean(
                tf.square(total_energy_pred - outputs))  # losses.MSE(total_energy, total_energy_pred)

        # Compute and apply gradients
        gradients = tape.gradient(total_energy_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_energy_loss


    @tf.function
    def val_step(inputs, outputs, p_energy, p_force):
        forces, total_energy = outputs

        # Convert forces to float32 to match the model output
        forces = tf.cast(forces, dtype=tf.float32)

        # Make predictions
        prediction = model([inputs[0], inputs[1], inputs[2], inputs[3]], training=False)

        # Separate energy and force predictions
        energy_pred = prediction[:, 0]
        force_pred = prediction[:, 1:]

        # Total energy is the sum of atomic energies
        total_energy_pred = tf.math.reduce_sum(energy_pred)

        # Calculate losses
        total_energy_loss = losses.MSE(total_energy, total_energy_pred)
        force_loss = losses.MSE(forces, force_pred)
        force_loss = tf.reduce_mean(force_loss)

        # Total loss is the sum of total energy loss and force loss
        total_loss = p_energy * total_energy_loss + p_force * force_loss

        return total_loss, total_energy_loss, force_loss

    # create the training directory
    checkpoint_dir = create_training_directory()

    train_ds, train_size = create_tf_dataset(os.path.join(Dataset.FOLDER, Dataset.TRAIN_NAME))
    val_ds, val_size = create_tf_dataset(os.path.join(Dataset.FOLDER, Dataset.VAL_NAME))
    model = get_model()

    # Optimizer with learning rate scheduler
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=Hyperparameters.initial_learning_rate,
        decay_steps=Hyperparameters.lr_decay_steps,
        decay_rate=Hyperparameters.decay_rate,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    # weightings for the loss function
    p_energy = Hyperparameters.p_energy_start
    p_force = Hyperparameters.p_force_start

    # counter for the steps
    step = 0

    # Training loop
    for epoch in range(Hyperparameters.EPOCHS):
        i = 0
        train_losses = []
        energy_losses = []
        force_losses = []
        for element in train_ds:
            x, y = element
            if Model.predict_only_energy:
                total_energy_loss = train_step_energy_only(
                    inputs=x,
                    outputs=y
                )
            else:
                training_loss, total_energy_loss, force_loss = train_step(
                    inputs=x,
                    outputs=y,
                    p_energy=tf.constant(p_energy, dtype=tf.float32),
                    p_force=tf.constant(p_force, dtype=tf.float32)
                )

            # training progress
            train_losses.append(training_loss if not Model.predict_only_energy else np.nan)
            energy_losses.append(total_energy_loss)
            force_losses.append(force_loss if not Model.predict_only_energy else np.nan)
            i += 1
            step += 1

            # update the weightings
            if not Model.predict_only_energy:
                p_energy = get_weighting_from_step(
                    step,
                    Hyperparameters.p_energy_limit,
                    Hyperparameters.p_energy_start,
                    Hyperparameters.decay_steps,
                    Hyperparameters.decay_rate
                )
                p_force = get_weighting_from_step(
                    step,
                    Hyperparameters.p_force_limit,
                    Hyperparameters.p_force_start,
                    Hyperparameters.decay_steps,
                    Hyperparameters.decay_rate
                )

            # Print mean loss every 1000 steps
            if i % 1000 == 0:
                print(f"Epoch {epoch} - Training step {i}/{train_size if train_size else 'Unknown'}")
                if not Model.predict_only_energy:
                    print(f"Total loss: {training_loss:.2f}")
                    print(f"Force loss: {force_loss:.2f} kcal mol^-1 Å^-1")
                print(f"Energy loss: {total_energy_loss:.2f} kcal mol^-1", end="\n\n")

        epoch_train_loss = np.mean(train_losses)
        epoch_force_loss = np.mean(force_losses)
        epoch_energy_loss = np.mean(energy_losses)

        # Validation
        epoch_val_loss = None
        epoch_val_energy_loss = None
        epoch_val_force_loss = None
        if val_ds is not None:
            print("Starting Validation")
            val_losses = np.zeros(val_size, dtype=np.float32)
            val_energy_losses = np.zeros(val_size, dtype=np.float32)
            val_force_losses = np.zeros(val_size, dtype=np.float32)
            for i, element in tqdm(enumerate(val_ds), total=val_size, desc="Validation", unit="batch"):
                x, y = element
                val_loss, val_energy_loss, val_force_loss = val_step(
                    inputs=x,
                    outputs=y,
                    p_energy=tf.constant(p_energy, dtype=tf.float32),
                    p_force=tf.constant(p_force, dtype=tf.float32)
                )
                val_losses[i] = val_loss.numpy()
                val_energy_losses[i] = val_energy_loss.numpy()
                val_force_losses[i] = val_force_loss.numpy()
            epoch_val_loss = np.mean(val_losses)
            epoch_val_energy_loss = np.mean(val_energy_losses)
            epoch_val_force_loss = np.mean(val_force_losses)
            print("Validation finished", end="\n\n")

        # log the training progress
        log_training_progress(
            epoch,
            train_loss=epoch_train_loss,
            train_energy_loss=epoch_energy_loss,
            train_force_loss=epoch_force_loss,
            p_energy=p_energy,
            p_force=p_force,
            val_loss=epoch_val_loss,
            val_energy_loss=epoch_val_energy_loss,
            val_force_loss=epoch_val_force_loss,
            directory=checkpoint_dir
        )

        # print training progress (one number after the comma)
        if not Model.predict_only_energy:
            print(f"Epoch {epoch} - p_energy: {p_energy}")
            print(f"Epoch {epoch} - p_force: {p_force}")
            print(f"Epoch {epoch} - Training loss: {epoch_train_loss:.2f}")
            print(f"Epoch {epoch} - Force loss: {epoch_force_loss:.2f} kcal mol^-1 Å^-1")
        print(f"Epoch {epoch} - Energy loss: {epoch_energy_loss:.2f} kcal mol^-1")
        if epoch_val_loss is not None:
            print(f"Epoch {epoch} - Validation loss: {epoch_val_loss:.2f}")
            print(f"Epoch {epoch} - Validation energy loss: {epoch_val_energy_loss:.2f} kcal mol^-1")
            print(f"Epoch {epoch} - Validation force loss: {epoch_val_force_loss:.2f} kcal mol^-1 Å^-1")

        # Save the model after each epoch
        model.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.h5"))
        print(f"Epoch {epoch} finished", end="\n\n")


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    start_training()
