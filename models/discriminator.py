import tensorflow as tf

def discriminator(config: dict,
                  name: str = "discriminator") -> tf.keras.Model:
    
    z_dim = config["z_dim"]
    kernel_init = config["kernel_init"]
    units = config["disc_units"]

    discriminator = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(z_dim,)),
            tf.keras.layers.Dense(units=units, 
                                    kernel_initializer=kernel_init,
                                    activation="relu"),
            tf.keras.layers.Dense(units=units, 
                                    kernel_initializer=kernel_init,
                                    activation="relu"),
            tf.keras.layers.Dense(units=units, 
                                    kernel_initializer=kernel_init,
                                    activation="relu"),
            tf.keras.layers.Dense(units=units, 
                                    kernel_initializer=kernel_init,
                                    activation="relu"),
            tf.keras.layers.Dense(units=units, 
                                    kernel_initializer=kernel_init,
                                    activation="relu"),
            tf.keras.layers.Dense(units=1, 
                                    kernel_initializer=kernel_init,
                                    activation="sigmoid")
        ], name=name)
    return discriminator

if __name__ == "__main__":

    from test_utils import fake_config
    from test_utils import check_shape_consistency_discrim, calc_num_parameters

    dis = discriminator(fake_config)

    # If you want a verbose description of the model
    # un-comment the following command.
    #dis.summary()

    calc_num_parameters(dis)

    check_shape_consistency_discrim(dis)