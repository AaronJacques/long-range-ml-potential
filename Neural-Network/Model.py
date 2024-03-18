import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, concatenate, Dense, Flatten, Lambda, Concatenate, Dropout, LayerNormalization, Add
)

from Constants import Model as ModelConstants


def local_embedding(local_matrix, atomic_numbers):
    # Extracting the first column using Lambda layer
    first_column_local_matrix = Lambda(lambda x: x[:, :, 0])(local_matrix)

    # insert the first column of local matrix into the atomic numbers matrix as a new column at the beginning
    first_column_local_matrix = tf.expand_dims(first_column_local_matrix, axis=-1)
    matrix = Concatenate(axis=-1)([first_column_local_matrix, atomic_numbers])

    # create embedding for the matrix with multiple Dense layers
    for dim in ModelConstants.embedding_dims:
        matrix = Dense(dim, activation='relu')(matrix)

    # TODO: add a Dropout layer here

    return matrix


def long_range_embedding(long_range_matrix, long_range_atomic_features):
    # Extracting the first column using Lambda layer
    first_column_long_range_matrix = Lambda(lambda x: x[:, :, 0])(long_range_matrix)

    # insert the first column of local matrix into the atomic features matrix as a new column at the beginning
    first_column_local_matrix = tf.expand_dims(first_column_long_range_matrix, axis=-1)
    matrix = Concatenate(axis=-1)([first_column_local_matrix, long_range_atomic_features])

    # create embedding for the matrix with multiple Dense layers
    for dim in ModelConstants.embedding_dims:
        matrix = Dense(dim, activation='relu')(matrix)

    # TODO: add a Dropout layer here

    return matrix


def feature_matrix(distance_matrix, embedding_matrix):
    g1 = embedding_matrix
    g2 = Lambda(lambda x: x[:, :, :ModelConstants.M2])(embedding_matrix)
    p1 = tf.linalg.matmul(distance_matrix, g2, transpose_a=True)
    p2 = tf.linalg.matmul(distance_matrix, p1)
    return tf.linalg.matmul(g1, p2, transpose_a=True)
    #return tf.matmul(tf.transpose(g1), tf.matmul(distance_matrix, tf.matmul(tf.transpose(distance_matrix), g2)))


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


def res_model(feature_matrix, return_shape):
    r = Dense(512, activation=ModelConstants.activation)(feature_matrix)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def res_o_shaped_model(feature_matrix, return_shape):
    r = Dense(256, activation=ModelConstants.activation)(feature_matrix)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 256, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def res_model_small(feature_matrix, return_shape):
    r = Dense(512, activation=ModelConstants.activation)(feature_matrix)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    r = dense_res_block(r, 512, activation=ModelConstants.activation)
    return Dense(return_shape, activation='linear')(r)


def get_model():
    input_local_matrix = Input(shape=ModelConstants.input_shape_local_matrix, name='input_local_matrix')
    input_atomic_numbers = Input(shape=ModelConstants.input_shape_atomic_numbers, name='input_atomic_numbers')
    input_long_range_matrix = Input(shape=ModelConstants.input_shape_long_range_matrix, name='input_long_range_matrix')
    input_long_range_atomic_features = Input(shape=ModelConstants.input_shape_long_range_atomic_features,
                                             name='input_long_range_atomic_features')

    # Local
    # embedding
    local_embedding_layer = local_embedding(input_local_matrix, input_atomic_numbers)
    # feature matrix
    local_feature_matrix = feature_matrix(input_local_matrix, local_embedding_layer)

    # Long range
    # embedding
    long_range_embedding_layer = long_range_embedding(input_long_range_matrix, input_long_range_atomic_features)
    # feature matrix
    long_range_feature_matrix = feature_matrix(input_long_range_matrix, long_range_embedding_layer)

    # concatenate and flatten the feature matrices
    concatenated_feature_matrix = Concatenate(axis=-1)([local_feature_matrix, long_range_feature_matrix])
    flattened_feature_matrix = Flatten()(concatenated_feature_matrix)

    # force network
    if ModelConstants.small_model:
        output = res_model_small(flattened_feature_matrix, return_shape=1 if ModelConstants.predict_only_energy else 4)
    else:
        output = res_model(flattened_feature_matrix, return_shape=1 if ModelConstants.predict_only_energy else 4)

    return Model(
        inputs=[input_local_matrix, input_atomic_numbers, input_long_range_matrix, input_long_range_atomic_features],
        outputs=output
    )


