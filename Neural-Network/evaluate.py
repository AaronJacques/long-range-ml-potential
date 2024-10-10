import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses
from tqdm import tqdm

from Dataset import create_tf_dataset

def evaluate_model(model_path, dataset_path, use_long_range=True):
    test_ds, test_size = create_tf_dataset(dataset_path)
    model = tf.keras.models.load_model(model_path)
    model.summary()

    @tf.function
    def test_step(inputs, outputs):
        # Make predictions
        prediction = model(
            [inputs[0], inputs[1], inputs[2], inputs[3]] if use_long_range else [inputs[0], inputs[1]],
            training=False
        )

        # Total energy is the sum of atomic energies
        total_energy_pred = tf.math.reduce_sum(prediction)

        # Calculate energy loss
        total_energy_loss = losses.MAE(outputs, total_energy_pred)

        return total_energy_loss

    print("Starting Evaluation")
    test_losses = np.zeros(test_size, dtype=np.float32)
    for i, element in tqdm(enumerate(test_ds), total=test_size, desc="Evaluation", unit="batch"):
        x, y = element
        test_loss = test_step(
            inputs=x,
            outputs=y
        )
        test_losses[i] = test_loss.numpy()

    test_loss = np.mean(test_losses)
    print("Evaluation finished", end="\n\n")
    print(f"Final MAE loss: {test_loss:.5f}")
    print("Evaluation finished")


if __name__ == "__main__":
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))
    evaluate_model(
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Paracetamol\Model-paracetamol_max_local_level_6_grid_size_1-DS-5e-05-lr-208000.0-decay-steps17-57-14_no_long_range\model_epoch_196.h5",
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Datasets\paracetamol_max_local_level_6_grid_size_1\test.pkl.gzip",
        False
    )
