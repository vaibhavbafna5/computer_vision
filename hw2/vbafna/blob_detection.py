import os

import numpy as np

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)

from filters import create_gaussian_kernel
from scipy.ndimage import convolve as cv


def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation
    # Input
    #   image: image of size HxW
    #   sigma: scalar standard deviation of Gaussian Kernel
    #
    # Output
    #   Gaussian filtered image of size HxW
    
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)

    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
        
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    output = cv(image, gaussian_kernel, mode='wrap')

    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # You can use your convolution function or scipy's convolution function
    return output


def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input
    #   image: image of size HxW
    #   min_sigma: smallest sigma in scale space
    #   k: scalar multiplier for scale space
    #   S: number of scales considers
    #
    # Output
    #   Scale Space of size HxWx(S-1)
    output = np.zeros((image.shape[0], image.shape[1], S - 1))
    
    sigmas = []
    for i in range(0, S):
        sigmas.append(min_sigma * (k ** i))

    for i in range(1, S):
        gauss_1 = gaussian_filter(image, sigmas[i - 1])
        gauss_2 = gaussian_filter(image, sigmas[i])
        output[:, :, i - 1] = (gauss_2 - gauss_1)

    return output


def main():
    image = read_img('polka.png')

    # Create directory for polka_detections
    if not os.path.exists("./polka_detections"):
        os.makedirs("./polka_detections")

    # -- Detecting Polka Dots
    print("Detect small polka dots")
    # -- Detect Small Circles
    sigma_1, sigma_2 = 11.5, 12.6
    gauss_1 = gaussian_filter(image, sigma_1)  # to implenent
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_small = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')

    # -- Detect Large Circles
    print("Detect large polka dots")
    sigma_1, sigma_2 = 53.5, 58.85
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # calculate difference of gaussians
    DoG_large = gauss_2 - gauss_1

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_large.png')

    # TODO Implement scale_space() and try to find both polka dots
    image = read_img('./polka.png')
    min_sigma = 11.5
    k=np.sqrt(2)
    S = 8

    res = scale_space(image=image, min_sigma=min_sigma, k=k, S=S)
    visualize_scale_space(res, min_sigma=11.5, k=np.sqrt(2), file_path='./scale_space.png')

    # TODO Detect the cells in any one (or more) image(s) from vgg_cells
    # Create directory for polka_detections
    if not os.path.exists("./cell_detections"):
        os.makedirs("./cell_detections")


if __name__ == '__main__':
    main()
