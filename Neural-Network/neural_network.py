import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Flatten, Lambda, Concatenate

from Constants import Model


def get_embedding(s, species1, species2, M1=32, embedding_size=10):
    num_distance_vectors = len(s)

    # Define the input layers
    input_atom1 = Input(shape=(1,), name='input_atom1')
    input_atom2 = Input(shape=(1,), name='input_atom2')
    input_distance = Input(shape=(1,), name='input_distance')

    # Repeat the input layers for each distance vector
    input_atom1_repeated = concatenate([input_atom1] * num_distance_vectors, axis=1)
    input_atom2_repeated = concatenate([input_atom2] * num_distance_vectors, axis=1)
    input_distance_vector = concatenate([input_distance] * num_distance_vectors, axis=1)

    # Define the embedding layers for atomic numbers
    embedding_layer_atom1 = Embedding(input_dim=100, output_dim=embedding_size)(input_atom1_repeated)
    embedding_layer_atom2 = Embedding(input_dim=100, output_dim=embedding_size)(input_atom2_repeated)

    # Flatten the embeddings
    flatten_layer_atom1 = Flatten()(embedding_layer_atom1)
    flatten_layer_atom2 = Flatten()(embedding_layer_atom2)

    # Concatenate the flattened embeddings with the distance input
    merged_layer = concatenate([flatten_layer_atom1, flatten_layer_atom2, input_distance_vector])

    # Define the dense layers for further processing
    dense_layer1 = Dense(4 * M1, activation='relu')(merged_layer)
    dense_layer2 = Dense(2 * M1, activation='relu')(dense_layer1)

    # Output layer
    return Dense(M1, activation='linear', name='output')(dense_layer2)


def local_embedding(local_matrix, atomic_numbers):
    # Extracting the first column using Lambda layer
    first_column_local_matrix = Lambda(lambda x: x[:, :, 0])(local_matrix)

    # concatenate the first column of local matrix with atomic numbers
    concatenated_input = Concatenate(axis=-1)([first_column_local_matrix, atomic_numbers])

    # Embedding layer
    return Embedding(input_dim=Model.embedding_size, output_dim=Model.M1)(concatenated_input)


def long_range_embedding(long_range_matrix, long_range_atomic_features):
    # Extracting the first column using Lambda layer
    first_column_long_range_matrix = Lambda(lambda x: x[:, :, 0])(long_range_matrix)

    # concatenate the first column of local matrix with atomic numbers
    concatenated_input = Concatenate(axis=-1)([first_column_long_range_matrix, long_range_atomic_features])

    # Embedding layer
    return Embedding(input_dim=Model.embedding_size, output_dim=Model.M1)(concatenated_input)


def feature_matrix(distance_matrix, embedding_matrix):
    g1 = embedding_matrix
    g2 = Lambda(lambda x: x[:, :Model.M2])(embedding_matrix)
    return tf.matmul(tf.transpose(g1), tf.matmul(distance_matrix, tf.matmul(tf.transpose(distance_matrix), g2)))


def force_network(feature_matrix):
    return Dense(3, activation='linear')(feature_matrix)


def model():
    input_local_matrix = Input(shape=Model.input_shape_local_matrix, name='input_local_matrix')
    input_atomic_numbers = Input(shape=Model.input_shape_atomic_numbers, name='input_atomic_numbers')
    input_long_range_matrix = Input(shape=Model.input_shape_long_range_matrix, name='input_long_range_matrix')
    input_long_range_atomic_features = Input(shape=Model.input_shape_long_range_atomic_features,
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
    force = force_network(flattened_feature_matrix)

    return Model(
        [input_local_matrix, input_atomic_numbers, input_long_range_matrix, input_long_range_atomic_features],
        force
    )


