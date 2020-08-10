import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras.layers import Flatten, Reshape


def normalization(norm_method: str = 'batch', **kwargs):
    if norm_method not in ['batch', 'layer']:
        raise ValueError
    if norm_method == 'batch':
        return BatchNormalization(**kwargs)
    elif norm_method == 'layer':
        return LayerNormalization(**kwargs)

def activation(func: str = 'relu', **kwargs):
    if func not in ['relu', 'leaky']:
        raise ValueError
    if func == 'relu':
        return ReLU(**kwargs)
    elif func == 'leaky':
        return LeakyReLU(**kwargs)

def encoder(config, 
            name: str = "func_encoder",
            norm_method: str = 'batch', 
            func: str = 'relu') -> tf.keras.Model:

    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["conv_kernel_size"]
    kernel_init = config["kernel_init"]
    
    inputs = Input(shape=(192, 84, 5))
    
    x = Conv2D(filters=h_dim, 
               kernel_size=kernel_size,
               use_bias=False,
               padding="same",
               kernel_initializer=kernel_init,
               strides=[2,1]) (inputs)
    x = normalization(norm_method=norm_method,
                      name='norm_1') (x)
    x = activation(func=func, name='act_1') (x)
    
    x = Conv2D(filters=2*h_dim, 
               kernel_size=kernel_size,
               use_bias=False,
               padding="same",
               kernel_initializer=kernel_init,
               strides=[2,2]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_2') (x)
    x = activation(func=func, name='act_2') (x)
    
    x = Conv2D(filters=4*h_dim, 
               kernel_size=kernel_size,
               use_bias=False,
               padding="same",
               kernel_initializer=kernel_init,
               strides=[2,2]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_3') (x)
    x = activation(func=func, name='act_3') (x)
    
    x = Conv2D(filters=8*h_dim, 
               kernel_size=kernel_size,
               use_bias=False,
               padding="same",
               kernel_initializer=kernel_init,
               strides=[3,3]) (x)
    x = normalization(norm_method=norm_method,
                      name='norm_4') (x)
    x = activation(func=func, name='act_4') (x)
    
    x = Flatten() (x)
    
    outputs = Dense(units=z_dim,
                    kernel_initializer=kernel_init) (x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def decoder(config, 
            name: str = "func_decoder",
            norm_method: str = 'batch', 
            func: str = 'relu') -> tf.keras.Model:

    h_dim = config["h_dim"]
    z_dim = config["z_dim"]
    kernel_size = config["conv_kernel_size"]
    kernel_init = config["kernel_init"]

    inputs = Input(shape=(z_dim))
    x = Dense(units=8*7*(8*h_dim),
              kernel_initializer=kernel_init) (inputs)
    #x = normalization(norm_method=norm_method) (x)
    x = activation(func=func) (x)

    x = Reshape((8, 7, 8*h_dim)) (x)

    x = Conv2DTranspose(filters=4*h_dim, 
                        kernel_size=kernel_size, 
                        use_bias=False,
                        padding="same",
                        kernel_initializer=kernel_init,
                        strides=[3,3]) (x)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func, name='act_1') (x)
    
    x = Conv2DTranspose(filters=2*h_dim, 
                        kernel_size=kernel_size,
                        use_bias=False,
                        padding="same",
                        kernel_initializer=kernel_init,
                        strides=[2,2]) (x)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func, name='act_2') (x)
    
    x = Conv2DTranspose(filters=h_dim, 
                        kernel_size=kernel_size,
                        use_bias=False,
                        padding="same",
                        kernel_initializer=kernel_init,
                        strides=[2,2]) (x)
    x = normalization(norm_method=norm_method) (x)
    x = activation(func=func, name='act_3') (x)
    
    outputs = Conv2DTranspose(filters=5, 
                              kernel_size=kernel_size, 
                              padding="same",
                              kernel_initializer=kernel_init,
                              strides=[2,1],
                              activation="sigmoid") (x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

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