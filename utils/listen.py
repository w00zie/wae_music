import numpy as np
from plot_utils import save_pianoroll, display_pianoroll
import os

def decode_dir(source_path: str, 
               dest_path: str, 
               thresholding: str = 'HD') -> None:
    """
    Turns `.npy` files into `.mid` files, converting
    the output tensors to MIDI.
    The default binarization method is `HD` (see Parameters).
    
    The expected behavious is:
    `source_path/tensor.npy` -> `dest_dir/tensor_HD.mid`
    or
    `source_path/tensor.npy` -> `dest_dir/tensor_BS.mid`

    Parameters
    ----------
    source_path : str 
        The directory containing the `.npy` files
        that you want to convert
        (e.g. `./runs/exp_000/samples/random_draws`).

    dest_path: str 
        The directory where you want to save
        your midi files (if does not exist gets created).
    
    thresholding: str
        The thresholding method. Valid choices are
            1. HD: Hard thresholding
            2. BS: Bernoulli sampling
        If not one of these two the thresholding
        method automatically becomes `HD`.
    """
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if thresholding not in ['HD', 'BS']:
        thresholding = 'HD'
    # Programs for midi conversion. 
    # See https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
    # Data ordering is [Drums, Piano, Guitar, Bass, String]
    programs = [0, 0, 25, 33, 48]
    files = [f for f in os.listdir(source_path) if f.endswith('.npy')]
    for f in files:
        name = "{}_{}.mid".format(f.split('.')[0], thresholding)
        data = np.load(os.path.join(source_path, f))
        data = data.reshape((-1, 4, 48, 84, 5))
        if thresholding == 'HD':
            data = (data > 0.5)
        else:
            data = (data > np.random.uniform(size=data.shape))
        # Making sure that the datatype is `bool`
        data = data.astype(np.bool)
        save_pianoroll(os.path.join(dest_path, name),
                       data,
                       programs,
                       is_drums=[True,False,False,False,False],
                       tempo=100,
                       beat_resolution=12,
                       lowest_pitch=24)

def listen_to_results(experiment_dir: str) -> None:
    """Goes through the `random_draws` and `reconstruction`
    subdirs of your `./runs/experiment` dir 
    and decodes them (if they exists) with `listen.decode_dir`.

    Parameters
    ----------
    experiment_dir: str
        The path of the experiment directory that you want to decode.
    """
    assert os.path.exists(experiment_dir), "Directory missing!"
    for subdir in ['random_draws', 'reconstructions']:
        source_dir = os.path.join(experiment_dir, subdir)
        if os.path.exists(source_dir):
            print(f"\tProducing midi outputs for {source_dir}")
            dest_dir = os.path.join(experiment_dir, "{}_midi".format(subdir))
            decode_dir(source_dir, dest_dir)
        else:
            print(f"{subdir} subdirectory missing!")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='./runs/wae_mmd_01_08_2020-09:47:39')
    args = parser.parse_args()

    print(f"Producing midi outputs for experiment {args.exp_dir}  ...")
    listen_to_results(os.path.join(args.exp_dir,"samples"))
    print("Done.")
