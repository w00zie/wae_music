import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll


def plot_pianoroll(
    ax,
    pianoroll,
    is_drum=False,
    beat_resolution=None,
    downbeats=None,
    preset="default",
    cmap="Blues",
    xtick="auto",
    ytick="octave",
    xticklabel=True,
    yticklabel="auto",
    tick_loc=None,
    tick_direction="in",
    label="both",
    grid="both",
    grid_linestyle=":",
    grid_linewidth=0.5,
):
    """Adapted from
    https://github.com/salu133445/pypianoroll/blob/master/pypianoroll/visualization.py#L25
    """
    if pianoroll.ndim not in (2, 3):
        raise ValueError("`pianoroll` must be a 2D or 3D numpy array")
    if xtick not in ("auto", "beat", "step", "off"):
        raise ValueError("`xtick` must be one of {'auto', 'beat', 'step', 'none'}.")
    if xtick == "beat" and beat_resolution is None:
        raise ValueError("`beat_resolution` must be specified when `xtick` is 'beat'.")
    if ytick not in ("octave", "pitch", "off"):
        raise ValueError("`ytick` must be one of {octave', 'pitch', 'off'}.")
    if not isinstance(xticklabel, bool):
        raise TypeError("`xticklabel` must be bool.")
    if yticklabel not in ("auto", "name", "number", "off"):
        raise ValueError(
            "`yticklabel` must be one of {'auto', 'name', 'number', 'off'}."
        )
    if tick_direction not in ("in", "out", "inout"):
        raise ValueError("`tick_direction` must be one of {'in', 'out', 'inout'}.")
    if label not in ("x", "y", "both", "off"):
        raise ValueError("`label` must be one of {'x', 'y', 'both', 'off'}.")
    if grid not in ("x", "y", "both", "off"):
        raise ValueError("`grid` must be one of {'x', 'y', 'both', 'off'}.")

    # plotting
    if pianoroll.ndim > 2:
        to_plot = pianoroll.transpose(1, 0, 2)
    else:
        to_plot = pianoroll.T
    if np.issubdtype(pianoroll.dtype, np.bool_) or np.issubdtype(
        pianoroll.dtype, np.floating
    ):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=1,
            origin="lower",
            interpolation="none",
        )
    elif np.issubdtype(pianoroll.dtype, np.integer):
        ax.imshow(
            to_plot,
            cmap=cmap,
            aspect="auto",
            vmin=0,
            vmax=127,
            origin="lower",
            interpolation="none",
        )
    else:
        raise TypeError("Unsupported data type for `pianoroll`.")

    # tick setting
    if tick_loc is None:
        tick_loc = ("bottom", "left")
    if xtick == "auto":
        xtick = "beat" if beat_resolution is not None else "step"
    if yticklabel == "auto":
        yticklabel = "name" if ytick == "octave" else "number"

    if preset == "plain":
        ax.axis("off")
    elif preset == "frame":
        ax.tick_params(
            direction=tick_direction,
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
    else:
        ax.tick_params(
            direction=tick_direction,
            bottom=("bottom" in tick_loc),
            top=("top" in tick_loc),
            left=("left" in tick_loc),
            right=("right" in tick_loc),
            labelbottom=(xticklabel != "off"),
            labelleft=(yticklabel != "off"),
            labeltop=False,
            labelright=False,
        )

    # x-axis
    if xtick == "beat" and preset != "frame":
        num_beat = pianoroll.shape[0] // beat_resolution
        ax.set_xticks(beat_resolution * np.arange(num_beat) - 0.5)
        ax.set_xticklabels("")
        ax.set_xticks(beat_resolution * (np.arange(num_beat) + 0.5) - 0.5, minor=True)
        ax.set_xticklabels(np.arange(1, num_beat + 1), minor=True)
        ax.tick_params(axis="x", which="minor", width=0)

    # y-axis
    if ytick == "octave":
        ax.set_yticks(np.arange(0, 84, 12))
        if yticklabel == "name":
            ax.set_yticklabels(["C{}".format(i - 2) for i in range(11)])
    elif ytick == "step":
        ax.set_yticks(np.arange(0, 84))
        if yticklabel == "name":
            if is_drum:
                ax.set_yticklabels(
                    [pretty_midi.note_number_to_drum_name(i) for i in range(84)]
                )
            else:
                ax.set_yticklabels(
                    [pretty_midi.note_number_to_name(i) for i in range(84)]
                )

    # axis labels
    if label in ("x", "both"):
        if xtick == "step" or not xticklabel:
            ax.set_xlabel("time (step)")
        else:
            ax.set_xlabel("time (beat)")

    if label in ("y", "both"):
        if is_drum:
            ax.set_ylabel("key name")
        else:
            ax.set_ylabel("pitch")

    # grid
    if grid != "off":
        ax.grid(
            axis=grid, color="k", linestyle=grid_linestyle, linewidth=grid_linewidth
        )

    # downbeat boarder
    if downbeats is not None and preset != "plain":
        for step in downbeats:
            ax.axvline(x=step, color="k", linewidth=1)

def save_pianoroll(filename, pianoroll, programs, is_drums, tempo,
                   beat_resolution, lowest_pitch):
    """Saves a batched pianoroll array to a npz file.
    Adapted from https://github.com/salu133445/musegan/blob/master/src/musegan/io_utils.py#L233
    """
    if not np.issubdtype(pianoroll.dtype, np.bool_):
        raise TypeError("Input pianoroll array must have a boolean dtype.")
    if pianoroll.ndim != 5:
        raise ValueError("Input pianoroll array must have 5 dimensions.")
    if pianoroll.shape[-1] != len(programs):
        raise ValueError("Length of `programs` does not match the number of "
                         "tracks for the input array.")
    if pianoroll.shape[-1] != len(is_drums):
        raise ValueError("Length of `is_drums` does not match the number of "
                         "tracks for the input array.")

    reshaped = pianoroll.reshape(
        -1, pianoroll.shape[1] * pianoroll.shape[2], pianoroll.shape[3],
        pianoroll.shape[4])

    # Pad to the correct pitch range and add silence between phrases
    to_pad_pitch_high = 128 - lowest_pitch - pianoroll.shape[3]
    padded = np.pad(
        reshaped, ((0, 0), (0, pianoroll.shape[2]),
                   (lowest_pitch, to_pad_pitch_high), (0, 0)), 'constant')

    # Reshape the batched pianoroll array to a single pianoroll array
    pianoroll_ = padded.reshape(-1, padded.shape[2], padded.shape[3])

    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    # Create and save the multitrack
    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
    multitrack.write(filename)
    #multitrack.save(filename)

def display_pianoroll(batch: tf.Tensor, path: str, cmap: str = 'Blues') -> None:
    if batch.shape[0] == 4:
        batch = batch.reshape(192, 84, 5)
    assert batch.shape == (192, 84, 5), 'Shape error in displaying pianoroll'
    instrum = [batch[:,:,i] for i in range(5)]
    fig, axes = plt.subplots(5, figsize=(11, 6))
    for inst, ax in zip(instrum, axes):
        plot_pianoroll(ax, inst, label='off', ytick='off', cmap=cmap)
    plt.savefig(path)

def get_matrix_to_be_plotted(x: tf.Tensor, num_cols: int = 8) -> tf.Tensor:
    """Stacks a tensor of (num_images, height, width) images to a matrix
    of shape (heigth * int(num_images / num_cols), height * num_cols)
    in order to being able to display it compactly.

                -----
             -----  |           ----- -----
          -----  |---           |    |    |
       -----  |---       ===>   ----- -----
       |   |---                 |    |    |  
       -----                    ----- -----

    """
    concat_x = lambda x,i: tf.concat([x[:,:,i] for i in range(num_cols*i, num_cols*(i+1))], axis=1)
    n_rows, leftovers = int(x.shape[-1] / num_cols), x.shape[-1] % num_cols
    assert leftovers == 0, 'Num cols not divisor of channels'
    assert n_rows != 0, 'Num rows not correct'
    mat = tf.concat([concat_x(x,i) for i in range(n_rows)], axis=0)
    return mat