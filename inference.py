import tensorflow as tf

import sys
import os
import json

from utils.get_model import get_ae

def sample_pz(batch_size=1, z_dim=16, sigma_z=1.) -> tf.Tensor:
        return tf.random.normal(shape=(batch_size, z_dim),
                                stddev=tf.sqrt(sigma_z))

def load_models(config: dict,
                enc_chkpt: str,
                dec_chkpt: str):

    enc, dec = get_ae(config=config, show_num_params=True)

    enc.load_weights(enc_chkpt)
    dec.load_weights(dec_chkpt)

    return (enc, dec)

def round_to_nearest(n, m):
    """Returns n rounded to the nearest multiple of m"""
    return m if n <= m else ((n / m) + 1) * m 

def get_first_untested():
    candidates = []
    root = "./runs"
    subdirs = [d for d in os.listdir(root) if \
        os.path.isdir(os.path.join(root, d)) and \
            'wae' in d]

    for d in sorted(subdirs):
        samples_dir = os.path.join(root, d, "samples")
        if not os.path.exists(samples_dir):
            candidates.append(os.path.join(root,d))
        else:
            continue

    if len(candidates) > 0:
        return candidates[0]
    else:
        print("\nAll experiments in `./runs` have been tested once.")
        print("\nArgument `next` can only be used when there is at least one untested dir.")
        print("Please specify the desired experiment to test (e.g. run `python inference.py --exp_dir ./runs/exp_000` )\n")
        exit()

if __name__ == "__main__":

    import os
    import json
    import argparse
    import numpy as np
    from utils.data import get_dataset
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    parser = argparse.ArgumentParser()
    # --exp_dir can either be:
    # 1. `next`: The experiment dir will be selected as the
    #            first (in lexicographical order) subdirectory 
    #            of `./runs/` that does not yet contain a `sample/` subdir.
    # 2. A path containing the desired subdirectory
    #    e.g. : `./runs/wae_gan_03_08_2020-08:15:21`           
    parser.add_argument('--exp_dir', type=str, default='next')
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--reconstruct', type=bool, default=True)
    args = parser.parse_args()

    logdir = args.exp_dir
    if logdir == 'next':
        logdir = get_first_untested()

    print(f'Loading config file from {logdir}')
    with open(os.path.join(logdir,"config.json")) as json_file:
        config = json.load(json_file)
    print('Done.\n')
    
    print('Loading network weights...')
    enc_chkpt = os.path.join(logdir,"models","encoder","encoder")
    dec_chkpt = os.path.join(logdir,"models","decoder","decoder")

    enc, dec = load_models(config,
                           enc_chkpt,
                           dec_chkpt)
    print('Done.\n')
    
    test_dataset = get_dataset(filename=config["loc_test_array"],
                               mmap=None,
                               batch_size=config["batch_size"],
                               shuffle=False,
                               test=True)

    samples_dir = os.path.join(logdir, "samples")
    rand_dir = os.path.join(samples_dir, "random_draws")
    recon_dir = os.path.join(samples_dir, "reconstructions")

    if not os.path.exists(samples_dir):
        print(f'Creating directory {samples_dir}')
        os.mkdir(samples_dir)
    if not os.path.exists(rand_dir):
        print(f'Creating directory {rand_dir}')
        os.mkdir(rand_dir)
    if args.reconstruct:
        if not os.path.exists(recon_dir):
            print(f'Creating directory {recon_dir}')
            os.mkdir(recon_dir)

    print(f'\nDecoding {args.num_samples} samples from the latent space...')
    for i in range(args.num_samples):
        z = sample_pz(1, z_dim=config["z_dim"], sigma_z=config["sigma_z"])
        music = dec(z, training=False)
        np.save(os.path.join(rand_dir, "{:03d}.npy".format(i)), music.numpy())
    print('Done.')

    if args.reconstruct:
        num_samples_from_ds = round_to_nearest(args.num_samples, config["batch_size"])
        num_batches = int(num_samples_from_ds / config["batch_size"])
        print(f'\nReconstructing {num_samples_from_ds} samples from the original distribution...')
        for b, batch in test_dataset.take(num_batches).enumerate():
            for s, sample in enumerate(batch):
                # Need a rank-4 Tensor
                sample = tf.expand_dims(sample, axis=0)
                q_z = enc(sample, training=False)
                music = dec(q_z, training=False)
                i = config["batch_size"] * b + s
                np.save(os.path.join(recon_dir, "{:03d}_true.npy".format(i)), sample.numpy())
                np.save(os.path.join(recon_dir, "{:03d}_fake.npy".format(i)), music.numpy())
        print('Done.')
