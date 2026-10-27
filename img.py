import numpy as np
from PIL import Image
import cv2

### BASIC FUNCTIONS

def crop_to_square(img_array, size=None):

    if size is None:
        size = min(img_array.shape[:2])

    return img_array[:size, :size]

def resample(img_array, factor=1):

    if factor==1:
        return img_array

    h, w = img_array.shape[:2]
    
    return cv2.resize(img_array,
                      (round(w / factor), round(h / factor)),
                      interpolation=cv2.INTER_AREA)


### LOADING FUNCTIONS

def image_to_float_array(image_path, factor=1, size=None):

    """
    Converts RGB image into binary numpy array.
    
    Args:
        image_path (str): path to the image
        factor (int): subsampling factor
        size (int): length of the side of the square (in pixels).
    
    Returns:
        np.array: float array (0. to 255.)
    """
        
    img = Image.open(image_path)

    # Converts image into gray levels.
    img_gray = img.convert("L")
    img_array = np.array(img_gray, dtype=float)
    img_cropped = crop_to_square(img_array, size=size)
    img_subsampled = resample(img_cropped, factor=factor)

    return img_subsampled


def image_to_binary_array(image_path, thresh=128, factor=1, size=None):

    """
    Converts RGB image into binary numpy array.
    
    Args:
        image_path (str): path to the image
        thresh (int): binarization thershold (0-255)
        factor (int): subsampling factor
        size (int): length of the side of the square (in pixels).
    
    Returns:
        np.array: binary array (0 or 1)
    """

    float_array = image_to_float_array(image_path, factor=factor, size=size)

    # Binarization
    binary_array = (float_array >= thresh).astype(bool)
    
    return binary_array


def image_to_density_array(image_path, factor=1, size=None):
    
    """
    Converts RGB image into float numpy array (between 0 and 1).

    Args:
        image_path (str): path to the image

    Returns:
        np.array: float array (between 0 and 1)
    """

    float_array = image_to_float_array(image_path, factor=factor, size=size)
    img_density = float_array / 255

    return img_density


### GENERATE FIELDS

def poisson_field(size=500, cloud_cover=0.5, save_path=None):
    """
    Generates a random field.

    Args:
        save_path (str): save path for the generated field
        size (int): side length of the square (in pixels)
        cloud_cover (float): between 0 and 1

    Returns:
        np.array: boolean array
    """

    img_array = np.random.random((size, size)) < cloud_cover

    # Saving image
    if save_path is not None:
        img = Image.fromarray(img_array)
        img.save(save_path)

    return img_array

def disk(size=8000, r=3000, save_path=None):

    img_array = np.zeros((size, size), dtype=bool)

    i0, j0 = size//2, size//2

    for i in range(size):
        for j in range(size):
            if np.sqrt((i - i0)**2 + (j-j0)**2) < r:
                img_array[i, j] = 1
    
    # Saving image
    if save_path is not None:
        img = Image.fromarray(img_array)
        img.save(save_path)

    return img_array