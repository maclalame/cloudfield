import numpy as np
from PIL import Image

def image_to_binary_array(image_path, threshold=128):
    """
    Converts RGB image into binary numpy array.
    
    Args:
        image_path (str): path to the image
        threshold (int): binarization thershold (0-255)
    
    Returns:
        np.array: binary array (0 or 1)
    """

    img = Image.open(image_path)
    
    # Converts image into gray levels.
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    # Binarization
    binary_array = (img_array >= threshold).astype(np.uint8)
    
    return binary_array


def image_to_density_array(image_path):
    """
    Converts RGB image into float numpy array (between 0 and 1).

    Args:
        image_path (str): path to the image

    Returns:
        np.array: float array (between 0 and 1)
    """

    img = Image.open(image_path)
    
    # Converts image into gray levels
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    img_density = img_array / 255

    return img_density


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