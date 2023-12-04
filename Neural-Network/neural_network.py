import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Flatten


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
