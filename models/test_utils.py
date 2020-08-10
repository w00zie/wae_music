import tensorflow as tf

from typing import List

# Fake hyperparameters used for testing modules functioning
fake_config = {
    "h_dim": 32,
    "z_dim": 128,
    "conv_kernel_size": [6,6],
    "kernel_init": "he_normal",
    "disc_units": 64,
}

def calc_num_parameters(model: tf.keras.Model):
    """
    Calculates and prints the number of trainable parameters
    in a tf.keras.Model.
    """
    tv = tf.reduce_sum([tf.reduce_prod(v.shape) for 
                        v in model.trainable_variables])
    tf.print(f"{model.name} has {tv} trainable params.")

def check_shape_consistency(encoder: tf.keras.Model,
                            decoder: tf.keras.Model,
                            shape: List = [32, 192, 84, 5],
                            module_name: str = __file__) -> None:
    batch = tf.random.uniform(shape=shape)
    z = tf.random.normal(shape=(1, fake_config["z_dim"]))
    z_hat = encoder(batch)
    x_hat = decoder(z_hat)

    print(f'{batch.shape} -> {z_hat.shape} -> {x_hat.shape}')
    assert batch.shape == x_hat.shape, f"Reconstruction shape error in {module_name}!"

def check_shape_consistency_discrim(discriminator: tf.keras.Model) -> None:
    bsize = 32
    batch = tf.random.uniform(shape=[bsize, fake_config["z_dim"]])
    q_z = discriminator(batch)

    print(f'{batch.shape} -> {q_z.shape}')
    assert q_z.shape == (bsize,1), f"Discriminator shape error!"