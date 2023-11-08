from layer import self_attention, cross_attention, feed_forward, encoder, decoder, transformer,CustomSchedule
import tensorflow as tf

import matplotlib.pyplot as plt

def plot(seq_length, key_dim, num_heads, ff_dim, vocab_size_en):
        
    model = self_attention(input_shape=(seq_length, key_dim),
                        num_heads=num_heads, key_dim=key_dim)
    tf.keras.utils.plot_model(model, "self-attention.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)

    model = cross_attention(input_shape=(seq_length, key_dim),
                            context_shape=(seq_length, key_dim),
                            num_heads=num_heads, key_dim=key_dim)
    tf.keras.utils.plot_model(model, "cross-attention.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)
    model = feed_forward(input_shape=(seq_length, key_dim),
                        model_dim=key_dim, ff_dim=ff_dim)
    tf.keras.utils.plot_model(model, "feedforward.png",
                            show_shapes=True, show_dtype=True, show_layer_names=True,
                            rankdir='BT', show_layer_activations=True)
    
    model = encoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                num_heads=num_heads)
    tf.keras.utils.plot_model(model, "encoder.png",
                          show_shapes=True, show_dtype=True, show_layer_names=True,
                          rankdir='BT', show_layer_activations=True)
    
    model = decoder(input_shape=(seq_length, key_dim), key_dim=key_dim, ff_dim=ff_dim,
                num_heads=num_heads)
    tf.keras.utils.plot_model(model, "decoder.png",
                          show_shapes=True, show_dtype=True, show_layer_names=True,
                          rankdir='BT', show_layer_activations=True)

    model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                    vocab_size_en, vocab_size_fr, dropout)
    tf.keras.utils.plot_model(model, "transformer.png",
                          show_shapes=True, show_dtype=True, show_layer_names=True,
                          rankdir='BT', show_layer_activations=True)
    
    lr = CustomSchedule(key_dim)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    plt.plot(lr(tf.range(50000, dtype=tf.float32)))



    plt.show()

    


seq_length = 20
key_dim = 128
num_heads = 8
ff_dim = 512
num_layers = 4
vocab_size_en = 14000
vocab_size_fr = 25000
seq_len = 20

dropout = 0.1

plot(seq_length, key_dim, num_heads, ff_dim, vocab_size_en)