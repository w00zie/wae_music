import tensorflow as tf

from models.model import encoder as fir_enc, decoder as fir_dec
from models.alt_model import encoder as alt_enc, decoder as alt_dec
from models.new_alt import encoder as new_alt_enc, decoder as new_alt_dec
from models.musegan_model import encoder as mus_enc, decoder as mus_dec
from models.newnew_alt import encoder as newnew_enc, decoder as newnew_dec
from models.discriminator import discriminator
from models.test_utils import calc_num_parameters

from typing import Tuple

def get_ae(config: dict,
           show_num_params: bool) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Returns a tuple with two Keras models (encoder, decoder) based
    on which `net_type` is specified in the `config` file.
    """
    net_type = config["net_type"]
    assert net_type in ["alt", 
                        "new_alt", 
                        "mus", 
                        "newnew",
                        "fir"], "Network type not implemented!"
    
    if net_type == "alt":
        enc, dec = alt_enc(config), alt_dec(config)
    elif net_type == "new_alt":
        enc, dec = new_alt_enc(config), new_alt_dec(config)
    elif net_type == "mus":
        enc, dec = mus_enc(config), mus_dec(config)
    elif net_type == "newnew":
        enc, dec = newnew_enc(config), newnew_dec(config)
    else:
        enc, dec = fir_enc(config), fir_dec(config)
    if show_num_params:
        calc_num_parameters(enc)
        calc_num_parameters(dec)
    return (enc, dec)

def get_discriminator(config: dict,
                      show_num_params: bool) -> tf.keras.Model:
    dis = discriminator(config)
    if show_num_params:
        calc_num_parameters(dis)
    return discriminator(config)

def get_ae_disc(config: dict,
                show_num_params: bool = True) -> Tuple[tf.keras.Model,
                                                       tf.keras.Model,
                                                       tf.keras.Model]:
    return (*get_ae(config, show_num_params), 
            get_discriminator(config, show_num_params)
    )

if __name__ == "__main__":

    from models.test_utils import fake_config
    from models.test_utils import check_shape_consistency

    for net_type in ["alt", "new_alt", "mus", "fir"]:
        print(f" ==== {net_type} ==== ")
        fake_config["net_type"] = net_type

        models = get_ae_disc(fake_config)
        assert len(models) == 3
    

    