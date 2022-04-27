import os

import numpy as np

from common import read_img, save_img

from filters import sobel_operator


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image.
    padding = max(window_size[0], window_size[1]) * 2
    image = np.pad(image, padding, 'constant')
    
    height = image.shape[0]
    width = image.shape[1]
    
    output = np.zeros(image.shape)

    total_sum = 0

    window_padding = int(np.floor(window_size[0] / 2))

    for y_pos in range(padding, height - padding):
        for x_pos in range(padding, width - padding):
            
            window1 = image[y_pos - window_padding : y_pos + window_padding + 1, x_pos - window_padding : x_pos + window_padding + 1]
#             print("y_pos: ", y_pos, "\n")
#             print("x_pos: ", x_pos, "\n")
#             print("window 1: \n", window1, "\n")
            
            offset_y_pos = y_pos + v
            offset_x_pos = x_pos + u

            window2 = image[offset_y_pos - window_padding : offset_y_pos + window_padding + 1, offset_x_pos - window_padding: offset_x_pos + window_padding + 1]
#             print("offset_y_pos: ", offset_y_pos, "\n")
#             print("offset_x_pos: ", offset_x_pos, "\n")
#             print("window 2: \n", window2, "\n")
            
            mid_sum = (window2 - window1) ** 2
            mid_sum = mid_sum.sum()
            output[y_pos, x_pos] = mid_sum

    return output


def harris_detector(image, window_size=(5, 5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    #
    # You can use same-padding for intensity (or 0-padding for derivatives)
    # to handle window values outside of the image.

    # compute the derivatives
    from scipy.ndimage import gaussian_filter

    Ix, Iy, _ = sobel_operator(image)

    Ixx = gaussian_filter(Ix ** 2, sigma=1)
    Ixy = gaussian_filter(Iy * Ix, sigma=1)
    Iyy = gaussian_filter(Iy ** 2, sigma=1)

    det = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    response = det - 0.05 * (trace) ** 2

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # Define offsets and window size and calulcate corner score
    u, v, W = 0, 5, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_1.png")

    u, v, W = 5, 0, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score_2.png")

    

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
