import numpy as np
import tensorflow as tf
import os
from functools import reduce

def load_data(filename: str, mmap) -> np.ndarray:
    assert os.path.isfile(filename), "File does not exists!"
    return np.load(filename, mmap_mode=mmap)

def data_generator(data_np_array: np.ndarray):
    return iter(data_np_array)

def get_dataset(filename: str, 
                batch_size: int = 64, 
                shuffle: bool = True,
                test: bool = False,
                mmap=None) -> tf.data.Dataset:
    data = load_data(filename, mmap=mmap)
    # Checking that the data comes in a 5-rank tensor format
    assert data.ndim == 5, "Data shape is not correct"
    assert data is not None, "Data not loaded correctly"
    # Checking that the following reshape is legal
    assert reduce(lambda a,b: a*b, data.shape[-4:]) == 192*84*5
    data = data.reshape(-1, 192, 84, 5)
    # Building a dataset from a generator, casting to tf.float32
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: data_generator(data),
        output_types=tf.float32,
        output_shapes=tf.TensorShape([192, 84, 5])
    )
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    if test:
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()

if __name__ == "__main__":
    from time import perf_counter
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    batch_size = 64
    train_set = get_dataset("../../dataset/lite_lpd5.npy", batch_size=batch_size, mmap=None)
    print('Train - with caching.')
    for i in range(3):
        start = perf_counter()
        for batch in train_set:
            assert batch is not None
            pass
        print(f'\t{i}: Took {perf_counter() - start}')
    print('Test - without caching.')
    test_set = get_dataset("../../dataset/lite_lpd5.npy", 
                           batch_size=batch_size, 
                           test=True,
                           mmap=None)
    for i in range(3):
        start = perf_counter()
        for batch in test_set:
            assert batch is not None
            pass
        print(f'\t{i}: Took {perf_counter() - start}')