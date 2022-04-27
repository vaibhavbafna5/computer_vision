import os

import numpy as np

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    # Given an input image and patch_size,
    # return the corresponding image patches made
    # by dividing up the image into patch_size sections.
    # Input- image: H x W
    #        patch_size: a scalar tuple M, N
    # Output- results: a list of images of size M x N

    # TODO: Use slicing to complete the function
    output = []

    patch_size_A = patch_size[0]
    patch_size_B = patch_size[1]

    beg_a = 0
    a = patch_size_A

    beg_b = 0
    b = patch_size_B

    print("a: ", a, "b: ", b)

    while a < 389:
        
        while b < 600:
            
            patch = image[beg_a:a, beg_b:b]
            patch = (patch - patch.mean()) / patch.std()
            output.append(patch)
            
            beg_b = b
            b += patch_size_B
            
        
        beg_a = a
        a += patch_size_A
        
        beg_b = 0
        b = patch_size_B

    return output


def conv_1d(im2d, ker1d):
    out = np.zeros(im2d.shape, dtype=im2d.dtype)  # allocate output assuming mode='same' 
    if ker1d.shape[0] == 1:
        # horizontal kernel
        for row in range(im2d.shape[0]):
            out[row,:] = np.convolve(im2d[row,:], ker1d.flatten(), mode='same')
    elif ker1d.shape[1] == 1:
        # vertical kernel
        for col in range(im2d.shape[1]):
            out[:,col] = np.convolve(im2d[:,col], ker1d.flatten(), mode='same')
    else:
        raise ValueError('input kernel is not 1D')
    return out

def convolve_regular(image, kernel):
    
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image
    
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]
            
    return output

def convolve(image, kernel):
    if kernel.shape[0] == 1 or kernel.shape[1] == 1:
        print("horizontal or vertical 1D kernel, use different function")
        return conv_1d(image, kernel)
    else:
        return convolve_regular(image, kernel)


def edge_detection(image):
    # Return the gradient magnitude of the input image
    # Input- image: H x W
    # Output- grad_magnitude: H x W

    # TODO: Use Ix, Iy to calculate grad_magnitude

    kx = np.array([[1/2, 0, -1/2]])
    ky = kx.transpose()
    
    x_out = convolve(image, kx)
    y_out = convolve(image, ky)
    
    res = np.hypot(x_out, y_out)
    return res, x_out, y_out


def sobel_operator(image):
    # Return Gx, Gy, and the gradient magnitude.
    # Input- image: H x W
    # Output- Gx, Gy, grad_magnitude: H x W

    # TODO: Use convolve() to complete the function
    Gx_kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    Gy_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])

    Gx = convolve(image, Gx_kernel)
    Gy = convolve(image, Gy_kernel)
    grad_magnitude = np.hypot(Gx, Gy)

    return Gx, Gy, grad_magnitude


def steerable_filter(image, angles=(np.pi * np.arange(6, dtype=np.float) / 6)):
    # Given a list of angels used as alpha in the formula,
    # return the corresponding images based on the formula given in pdf.
    # Input- image: H x W
    #        angels: a list of scalars
    # Output- results: a list of images of H x W

    # TODO: Use convolve() to complete the function
    output = []
    
    Gx_kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    Gy_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ])
    
    for angle in angles:
        K_kernel = (np.cos(angle) * Gx_kernel) + (np.sin(angle) * Gy_kernel)
        res = convolve(image, K_kernel)
        output.append(res)

    return output

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def create_gaussian_kernel(size=3, sigma=(1 / (2 * np.log(2)))):
    
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    
    return kernel_2D   


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- Image Patches --
    # Q1
    patches = image_patches(img)
    # TODO choose a few patches and save them
    chosen_patches = patches[815]
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # Q2: No code
    """ Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- Gaussian Filter --
    # Q1: No code

    # Q2

    # TODO: Calculate the kernel described in the question.
    # There is tolerance for the kernel.
    kernel_gaussian = create_gaussian_kernel()
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # # Q3
    edge_detect, _, _ = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    edge_with_gaussian, _, _ = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # -- Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Q1: No code

    # Q2
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    # Q3
    steerable_list = steerable_filter(img)
    for i, steerable in enumerate(steerable_list):
        save_img(steerable, "./sobel_operator/q3_steerable_{}.png".format(i))

    print("Sobel Operator is done. ")
    """ LoG Filter """
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # Q1
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # Q2: No code
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
