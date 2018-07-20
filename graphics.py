import numpy as np
from PIL import Image
import time
import threading


def save_image(x, path):
    im = Image.fromarray(x)
    im.save(path, optimize=True)
    return

# Assumes [NCHW] format
def save_raster(x, path, rescale=False, width=None):
    t = threading.Thread(target=_save_raster, args=(x, path, rescale, width))
    t.start()


def _save_raster(x, path, rescale, width):
    x = to_raster(x, rescale, width)
    save_image(x, path)

# Shape: (n_patches,rows,columns,channels)
def to_raster_old(x, rescale=False, width=None):
    x = np.transpose(x, (0, 3, 1, 2))

    #x = x.swapaxes(2, 3)
    if len(x.shape) == 3:
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    if x.shape[1] == 1:
        x = np.repeat(x, 3, axis=1)
    if rescale:
        x = (x - x.min()) / (x.max() - x.min()) * 255.
    x = np.clip(x, 0, 255)
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    n_patches = x.shape[0]
    if width is None:
        width = int(np.ceil(np.sqrt(n_patches)))  # result width
    height = int(n_patches/width)  # result height
    tile_height = x.shape[2]
    tile_width = x.shape[3]
    result = np.zeros((3, int(height*tile_height),
                       int(width*tile_width)), dtype='uint8')
    for i in range(height):
        for j in range(width):
            result[:, i*tile_height:(i+1)*tile_height,
                   j*tile_width:(j+1)*tile_width] = x[i]
    return result


# Shape: (n_patches,rows,columns,channels)
def to_raster(x, rescale=False, width=None):
    if len(x.shape) == 3:
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    if x.shape[3] == 1:
        x = np.repeat(x, 3, axis=3)
    if rescale:
        x = (x - x.min()) / (x.max() - x.min()) * 255.
    x = np.clip(x, 0, 255)
    assert len(x.shape) == 4
    assert x.shape[3] == 3
    n_batch = x.shape[0]
    if width is None:
        width = int(np.ceil(np.sqrt(n_batch)))  # result width
    height = int(n_batch / width)  # result height
    tile_height = x.shape[1]
    tile_width = x.shape[2]
    result = np.zeros((int(height * tile_height),
                       int(width * tile_width), 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            result[i * tile_height:(i + 1) * tile_height, j *
                   tile_width:(j + 1) * tile_width] = x[width*i+j]
    return result
