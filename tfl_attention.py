try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt

    print("skimage imports:")
    from skimage.feature import peak_local_max

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


#==============Auxiliary Functions==================

def find_lights(image: np.ndarray, kernel:np.ndarray):

    result = sg.convolve2d(image, kernel)
    max_dots = peak_local_max(result, min_distance=20, num_peaks=10)

    lights_dots = []
    for i in max_dots:
        lights_dots.append([i[1], i[0]])

    return lights_dots

#===================================================

def find_tfl_lights(image_path):

    image = np.array(Image.open(image_path))

    kernel_r = np.array(
        [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [1, 1, -0.5], [1, 2, 1],
         [1, 2, 1], [1, 1, 1], [1, 1, 1]])

    kernel_g = np.array(
        [[1, 1, 1], [1, 1, 1], [1, 2, 1], [1, 2, 1], [1, 1, -0.5], [1, 1, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
         [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], ])


    red_dots = find_lights(image[:, :, 0],kernel_r)
    green_dots = find_lights(image[:, :, 1], kernel_g)

    colors = ['red'] * len(red_dots) + ['green'] * len(green_dots)

    return red_dots + green_dots, colors