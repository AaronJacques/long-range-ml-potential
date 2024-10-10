import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, concatenate, Dense, Flatten, Lambda, Concatenate, Dropout, LayerNormalization, Add, Reshape
)

from Constants import Model as ModelConstants


def feature_matrix(distance_matrix, embedding_matrix, M2):
    g1 = embedding_matrix
    g2 = Lambda(lambda x: x[:, :, :M2])(embedding_matrix)
    p1 = tf.linalg.matmul(distance_matrix, g2, transpose_a=True)
    p2 = tf.linalg.matmul(distance_matrix, p1)
    return tf.linalg.matmul(g1, p2, transpose_a=True)


def dense_res_block(input, units, activation='relu'):
    r = LayerNormalization()(input)
    r = Dense(units, activation=activation)(r)
    r = Dense(units, activation=activation)(r)
    # dense layer to get correct shape for add
    r = Dense(int(input.get_shape().as_list()[1]), activation=activation)(r)
    r = Dropout(0.2)(r)
    # add the input to the output of the Dense layer
    r = Add()([r, input])
    r = LayerNormalization()(r)
    return r


def res_embedding_model(matrix, features, n_max, hyperparameters, M1):
    # Extracting the first column using Lambda layer
    first_column_local_matrix = Lambda(lambda x: x[:, :, 0])(matrix)

    # insert the first column of local matrix into the atomic numbers matrix as a new column at the beginning
    #first_column_local_matrix = tf.expand_dims(first_column_local_matrix, axis=-1)
    #matrix = Concatenate(axis=-1)([first_column_local_matrix, features])
    #matrix = Flatten()(matrix)
    features = Flatten()(features)
    matrix = Concatenate()([first_column_local_matrix, features])

    # create embedding for the matrix with multiple res blocks
    r = Dense(hyperparameters.embedding_dims[0], activation=ModelConstants.activation)(matrix)
    for dim in hyperparameters.embedding_dims[1:]:
        r = dense_res_block(r, dim, activation=ModelConstants.activation)

    # reshape the output to the correct shape
    r = Dense(n_max * M1, activation=ModelConstants.activation)(r)
    r = Reshape((n_max, M1))(r)

    return r


def res_model(feature_matrix, return_shape):
    r = Dense(128, activation=ModelConstants.activation)(feature_matrix)
    r = Dropout(0.2)(r)
    r = dense_res_block(r, 128, activation=ModelConstants.activation)
    r = dense_res_block(r, 128, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def res_o_shaped_model(feature_matrix, return_shape):
    r = Dense(256, activation=ModelConstants.activation)(feature_matrix)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def res_model_deep(feature_matrix, return_shape):
    r = Dense(512, activation=ModelConstants.activation)(feature_matrix)
    r = Dropout(0.2)(r)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def get_local_only_model(hyperparameters):
    input_local_matrix = Input(shape=ModelConstants.input_shape_local_matrix, name='input_local_matrix')
    input_atomic_numbers = Input(shape=ModelConstants.input_shape_atomic_numbers, name='input_atomic_numbers')

    # embedding
    local_embedding_layer = res_embedding_model(
        input_local_matrix,
        input_atomic_numbers,
        ModelConstants.n_max_local,
        hyperparameters,
        M1=hyperparameters.M1_local
    )
    # feature matrix
    local_feature_matrix = feature_matrix(
        input_local_matrix,
        local_embedding_layer,
        M2=hyperparameters.M2_local
    )

    # concatenate and flatten the feature matrices
    flattened_feature_matrix = Flatten()(local_feature_matrix)

    output = res_model(flattened_feature_matrix, return_shape=1 if ModelConstants.predict_only_energy else 4)

    return Model(
        inputs=[input_local_matrix, input_atomic_numbers],
        outputs=output
    )


def get_long_and_local_model(hyperparameters):
    input_local_matrix = Input(shape=ModelConstants.input_shape_local_matrix, name='input_local_matrix')
    input_atomic_numbers = Input(shape=ModelConstants.input_shape_atomic_numbers, name='input_atomic_numbers')
    input_long_range_matrix = Input(shape=ModelConstants.input_shape_long_range_matrix, name='input_long_range_matrix')
    input_long_range_atomic_features = Input(shape=ModelConstants.input_shape_long_range_atomic_features,
                                             name='input_long_range_atomic_features')

    # Local
    # embedding
    local_embedding_layer = res_embedding_model(
        input_local_matrix,
        input_atomic_numbers,
        ModelConstants.n_max_local,
        hyperparameters,
        M1=hyperparameters.M1_local
    )
    # feature matrix
    local_feature_matrix = feature_matrix(
        input_local_matrix,
        local_embedding_layer,
        M2=hyperparameters.M2_local
    )

    # Long range
    # embedding
    long_range_embedding_layer = res_embedding_model(
        input_long_range_matrix,
        input_long_range_atomic_features,
        ModelConstants.n_max_long_range,
        hyperparameters,
        M1=hyperparameters.M1_long
    )
    # feature matrix
    long_range_feature_matrix = feature_matrix(
        input_long_range_matrix,
        long_range_embedding_layer,
        M2=hyperparameters.M2_long
    )

    # concatenate and flatten the feature matrices
    if hyperparameters.M1_local == hyperparameters.M1_long and hyperparameters.M2_local == hyperparameters.M2_long:
        concatenated_feature_matrix = Concatenate(axis=-1)([local_feature_matrix, long_range_feature_matrix])
        reduced_feature_matrix = Dense(
            hyperparameters.M2_local, activation=ModelConstants.activation
        )(concatenated_feature_matrix)
        flattened_feature_matrix = Flatten()(reduced_feature_matrix)
    else:
        flattened_local_feature_matrix = Flatten()(local_feature_matrix)
        flattened_long_range_feature_matrix = Flatten()(long_range_feature_matrix)
        flattened_feature_matrix = Concatenate(axis=-1)(
            [flattened_local_feature_matrix, flattened_long_range_feature_matrix]
        )

    output = res_model(flattened_feature_matrix, return_shape=1 if ModelConstants.predict_only_energy else 4)
    # output = res_model_deep(flattened_feature_matrix, return_shape=1 if ModelConstants.predict_only_energy else 4)

    return Model(
        inputs=[input_local_matrix, input_atomic_numbers, input_long_range_matrix, input_long_range_atomic_features],
        outputs=output
    )


def get_model(hyperparameters):
    if ModelConstants.use_long_range:
        return get_long_and_local_model(hyperparameters)
    else:
        return get_local_only_model(hyperparameters)
