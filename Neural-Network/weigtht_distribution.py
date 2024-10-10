import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Function to extract and flatten weights from the model
def get_flattened_weights(model_path, layer_names):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    weights = []
    for layer in model.layers:
        if layer_names is None or layer.name in layer_names:
            layer_weights = layer.get_weights()
            for weight in layer_weights:
                weights.append(weight.flatten())
    # Flatten all weights into a single array
    all_weights = [w for weight_array in weights for w in weight_array]
    return all_weights


def plot_weight_distribution_comparison(
        model_path_with_long_range, model_path_without_long_range, molecule_name, title,
        layer_names=None, density=True,
):
    # Get the flattened weights for both models
    weights_with_long_range = get_flattened_weights(model_path_with_long_range, layer_names)
    weights_without_long_range = get_flattened_weights(model_path_without_long_range, layer_names)

    # Determine the y-axis limit across both histograms
    _, bins1, _ = plt.hist(weights_with_long_range, bins=50, density=density)
    _, bins2, _ = plt.hist(weights_without_long_range, bins=50, density=density)

    plt.close()

    # Plotting
    plt.figure(figsize=(12, 6))

    # We use the larger of the two bin ranges for consistency
    y_max1 = np.histogram(weights_with_long_range, bins=bins1, density=density)[0].max()
    y_max2 = np.histogram(weights_without_long_range, bins=bins2, density=density)[0].max()
    y_max = max(y_max1, y_max2) * 1.1

    # Subplot 1: With long-range interaction
    plt.subplot(1, 2, 1)
    plt.hist(weights_with_long_range, bins=50, color='blue', alpha=0.7, density=density)
    plt.ylim(0, y_max)
    plt.title('With Long-Range Interaction')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')

    # Subplot 2: Without long-range interaction
    plt.subplot(1, 2, 2)
    plt.hist(weights_without_long_range, bins=50, color='green', alpha=0.7, density=density)
    plt.ylim(0, y_max)
    plt.title('Without Long-Range Interaction')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')

    plt.suptitle(title + ' ' + molecule_name)
    plt.show()


def plot_stacked_weight_distribution(
        model_path_with_long_range, model_path_without_long_range, molecule_name, title, boundary=None,
        layer_names=None, density=True, save_name=None,
):
    # Get the flattened weights for both models
    weights_with_long_range = get_flattened_weights(model_path_with_long_range, layer_names)
    weights_without_long_range = get_flattened_weights(model_path_without_long_range, layer_names)
    print(f"Total number of weights with long-range interaction: {len(weights_with_long_range)}")
    print(f"Total number of weights without long-range interaction: {len(weights_without_long_range)}")

    # only take the weights between boundaries
    if boundary is not None:
        weights_with_long_range = [w for w in weights_with_long_range if boundary[0] < w < boundary[1]]
        weights_without_long_range = [w for w in weights_without_long_range if boundary[0] < w < boundary[1]]

    # Plotting
    plt.figure(figsize=(6, 5.5))
    plt.hist(
        [weights_with_long_range, weights_without_long_range],
        bins=150,
        density=density,
        histtype='step',
        label=['With', 'Without'],
    )

    # x-axis limits
    if boundary is not None:
        plt.xlim(boundary)
    font_size = 16
    plt.legend(loc="upper right", fontsize=font_size)
    plt.title(title + ' ' + molecule_name, fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('Weight Value', fontsize=font_size)
    plt.ylabel('Density', fontsize=font_size)
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=4)
    if save_name is not None:
        plt.savefig(save_name + '_' + molecule_name + '.pdf')
    else:
        plt.savefig(title + ' ' + molecule_name + '.pdf')
    plt.show()


local_embedding_layer_names = [
    "dense",
    "dense_1",
    "dense_2",
    "dense_3",
    "layer_normalization_1",
    "layer_normalization_2",
    "dense"
]


if __name__ == "__main__":
    title = "Weight Distribution"
    save_name = "weight_distribution"

    plot_stacked_weight_distribution(
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Paracetamol\Model-paracetamol_max_local_level_6_grid_size_1-DS-5e-05-lr-208000.0-decay-steps23-21-36\model_epoch_188.h5",
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Paracetamol\Model-paracetamol_max_local_level_6_grid_size_1-DS-5e-05-lr-208000.0-decay-steps17-57-14_no_long_range\model_epoch_196.h5",
        "Paracetamol",
        title=title,
        boundary=(-1, 1),
        save_name=save_name,
    )
    plot_stacked_weight_distribution(
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Aspirin\1. Run\Model-aspirin_max_local_level_6_grid_size_1-DS-0.0001-lr-208000.0-decay-steps19-56-15\model_epoch_189.h5",
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Aspirin\1. Run\Model-aspirin_max_local_level_6_grid_size_1-DS-0.0001-lr-208000.0-decay-steps08-28-24_no_long_range\model_epoch_193.h5",
        "Aspirin",
        title=title,
        boundary=(-1, 1),
        save_name=save_name,
    )
    plot_stacked_weight_distribution(
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Ac\Model-Ac-Ala3-NHMe_max_local_level_2_grid_size_2-DS-5e-05-lr-208000.0-decay-steps13-40-55\model_epoch_116.h5",
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Ac\Model-Ac-Ala3-NHMe_max_local_level_2_grid_size_2-DS-5e-05-lr-208000.0-decay-steps13-40-55\model_epoch_131.h5",
        "Ac-Ala3-NHMe",
        title=title,
        boundary=(-1, 1),
        save_name=save_name,
    )
    plot_stacked_weight_distribution(
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Stachyose\Model-stachyose_max_local_level_4_grid_size_2-DS-5e-05-lr-87270.0-decay-steps11-01-04\model_epoch_127.h5",
        r"D:\Uni\Master Physik\Masterarbeit\Repo\long-range-ml-potential\Checkpoints\Stachyose\Model-stachyose_max_local_level_4_grid_size_2-DS-5e-05-lr-87270.0-decay-steps20-54-08_no_long_range\model_epoch_117.h5",
        "Stachyose",
        title=title,
        boundary=(-1, 1),
        save_name=save_name,
    )
