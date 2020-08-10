import tensorflow as tf

def encoder(config: dict, name: str ="newnew_encoder") -> tf.keras.Model:

    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["conv_kernel_size"]
    kernel_init = config["kernel_init"]

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(192, 84, 5)),
            tf.keras.layers.Conv2D(filters=h_dim, 
                                    kernel_size=[12, 12],
                                    use_bias=False,
                                    strides=(2,1),
                                    kernel_initializer=kernel_init,
                                    padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_1'),
            tf.keras.layers.Conv2D(filters=2*h_dim, 
                                    kernel_size=[8,8],
                                    use_bias=False,
                                    strides=(2,2),
                                    kernel_initializer=kernel_init,
                                    padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_2'),
            tf.keras.layers.Conv2D(filters=4*h_dim, 
                                    kernel_size=[6,6],
                                    use_bias=False,
                                    strides=(2,2),
                                    kernel_initializer=kernel_init,
                                    padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_3'),
            tf.keras.layers.Conv2D(filters=8*h_dim, 
                                    kernel_size=[6,6],
                                    use_bias=False,
                                    strides=(3,3),
                                    kernel_initializer=kernel_init,
                                    padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_4'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=z_dim, 
                                  kernel_initializer=kernel_init)
        ], name=name)
    return encoder

def decoder(config: dict, name: str = "newnew_decoder") -> tf.keras.Model:
    
    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["conv_kernel_size"]
    kernel_init = config["kernel_init"]

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(z_dim,)),
            tf.keras.layers.Dense(units=8*7*(8*h_dim), 
                                    kernel_initializer=kernel_init),
            #tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((8, 7, 8*h_dim)),
            tf.keras.layers.Conv2DTranspose(filters=4*h_dim, 
                                            kernel_size=[6,6],
                                            use_bias=False,
                                            strides=(3,3),
                                            padding="same",
                                            kernel_initializer=kernel_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_1'),
            tf.keras.layers.Conv2DTranspose(filters=2*h_dim, 
                                            kernel_size=[6,6],
                                            use_bias=False,
                                            strides=(2,2),
                                            padding="same",
                                            kernel_initializer=kernel_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_2'),
            tf.keras.layers.Conv2DTranspose(filters=h_dim, 
                                            kernel_size=[8,8],
                                            use_bias=False,
                                            strides=(2,2),
                                            padding="same",
                                            kernel_initializer=kernel_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(name='act_3'),
            tf.keras.layers.Conv2DTranspose(filters=5, 
                                    kernel_size=[12, 12],
                                    strides=(2,1),
                                    padding="same",
                                    kernel_initializer=kernel_init,
                                    activation="sigmoid"),
        ], name=name)
    return decoder

if __name__ == "__main__":
    
    from test_utils import fake_config
    from test_utils import check_shape_consistency, calc_num_parameters

    encoder = encoder(fake_config)
    decoder = decoder(fake_config)

    # If you want a verbose description of the model
    # un-comment the following two commands.
    #encoder.summary()
    #decoder.summary()

    calc_num_parameters(encoder)
    calc_num_parameters(decoder)

    check_shape_consistency(encoder, decoder, module_name=__file__)