from tensorflow.keras import layers


def get_embedding_layer(vocab_size, embed_dim):
    embedding_layer = layers.Embedding(input_dim=vocab_size, 
                                    output_dim=embed_dim,
                                    embeddings_initializer='normal')
    return embedding_layer