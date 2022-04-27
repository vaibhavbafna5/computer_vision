"""
Starter code for EECS 442 W20 HW1
"""
from util import *
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2lab


def rotX(theta):
    # TODO: Return the rotation matrix for angle theta around X axis
    # calculate values
    cos_value = math.cos(theta)
    sin_value = math.sin(theta)
    neg_sin_value = -1 * sin_value

    # initialize rotation matrix
    x_rot_matrix = np.eye(3)

    # set cosine values
    x_rot_matrix[1][1] = cos_value
    x_rot_matrix[2][2] = cos_value

    # set sin values
    x_rot_matrix[1][2] = neg_sin_value
    x_rot_matrix[2][1] = sin_value

    return x_rot_matrix


def rotY(theta):
    # TODO: Return the rotation matrix for angle theta around Y axis
    # calculate values
    cos_value = math.cos(theta)
    sin_value = math.sin(theta)
    neg_sin_value = -1 * sin_value

    # initialize rotation matrix
    y_rot_matrix = np.eye(3)

    # set cosine values
    y_rot_matrix[0][0] = cos_value
    y_rot_matrix[2][2] = cos_value

    # set sin values
    y_rot_matrix[2][0] = neg_sin_value
    y_rot_matrix[0][2] = sin_value

    return y_rot_matrix


def part1():
    # generate gif
    rot = 0
    R = []
    while rot != math.pi * 2:
        rot += math.pi / 4
        R.append(rotY(rot))
    generate_gif(R)

    # problem 1B

    renderCube(R=rotY(np.pi / 4).dot(rotX(np.pi / 4)), file_name="take1")
    renderCube(R=rotX(np.pi / 4).dot(rotY(np.pi / 4)), file_name="take2")

    # problem 1C
    renderCube(R=rotX(math.pi / 5).dot(rotY(np.pi / 4)), file_name="1c")

    # problem 1D
    renderCube(f=np.inf, R=rotX(math.pi / 5).dot(rotY(np.pi / 4)), file_name="1d")


def split_triptych(trip):
    # TODO: Split a triptych into thirds and return three channels in numpy arrays
    img = trip
    height = int(img.shape[0] / 3)
    width = img.shape[1]

    img_1 = img[0:height][0:width]
    img_2 = img[height : 2 * height][0:width]
    img_3 = img[2 * height : 3 * height][0:width]

    stacked_imgs = [img_3, img_2, img_1]
    stacked_img = np.dstack(stacked_imgs)

    return stacked_img


def split_triptych_augmented(trip, img_data):

    y_start = img_data["y_start"]
    y_offset = img_data["y_offset"]
    x_start = img_data["x_start"]
    x_offset = img_data["x_offset"]

    stacks = []

    for i in range(0, 3):
        img_slice = trip[y_start : y_start + y_offset, x_start : x_start + x_offset]
        stacks.insert(0, img_slice)
        y_start += y_offset

    colored = np.dstack(stacks)
    return colored


def normalized_cross_correlation(p1, p2):
    product = np.mean((p1 - p1.mean()) * (p2 - p2.mean()))
    std = p1.std() * p2.std()
    if std != 0:
        res = product / std
        return res
    else:
        return 0


def normalized_cross_correlation_without_normalization(p1, p2):
    product = np.mean((p1) * (p2))
    std = p1.std() * p2.std()
    if std != 0:
        res = product
        return res
    else:
        return 0


def best_offset(ch1, ch2, metric, Xrange=np.arange(-15, 16), Yrange=np.arange(-15, 16)):

    best_offset = -1
    out = [0, 0]

    for x in Xrange:
        for y in Yrange:
            curr_offset = metric(ch1, np.roll(ch2, [x, y], axis=(0, 1)))
            if curr_offset > best_offset:
                best_offset = curr_offset
                out = [x, y]

    return out


def align_and_combine(R, G, B, metric):
    # TODO: Use metric to align three channels and return the combined RGB image
    r = R
    g = G
    b = B

    b_r = best_offset(R, B, metric)
    g_r = best_offset(R, G, metric)
    print("b --> r: ", b_r)
    print("g --> r: ", g_r)

    b = np.roll(b, b_r, axis=(0, 1))
    g = np.roll(g, g_r, axis=(0, 1))

    col = np.dstack([r, g, b])

    return col


def part2():

    # part 2.1 - combine image
    img = imageio.imread("prokudin-gorskii/00125v.jpg")
    stacked_img = split_triptych(img)
    imageio.imsave("combined.jpg", stacked_img)

    # part 2.2 - align image
    data = {
        "prokudin-gorskii/00125v.jpg": {
            "y_start": 28,
            "y_offset": 330,
            "x_start": 28,
            "x_offset": 388,
        },
        "prokudin-gorskii/00149v.jpg": {
            "y_start": 12,
            "y_offset": 330,
            "x_start": 28,
            "x_offset": 387,
        },
        "prokudin-gorskii/00153v.jpg": {
            "y_start": 12,
            "y_offset": 334,
            "x_start": 26,
            "x_offset": 380,
        },
        "prokudin-gorskii/00351v.jpg": {
            "y_start": 15,
            "y_offset": 334,
            "x_start": 25,
            "x_offset": 378,
        },
        "prokudin-gorskii/00398v.jpg": {
            "y_start": 26,
            "y_offset": 332,
            "x_start": 15,
            "x_offset": 378,
        },
        "prokudin-gorskii/01112v.jpg": {
            "y_start": 15,
            "y_offset": 332,
            "x_start": 14,
            "x_offset": 378,
        },
        "tableau/efros_tableau.jpg": {
            "y_start": 0,
            "y_offset": 418,
            "x_start": 0,
            "x_offset": 505,
        },
    }

    i = 0
    for file in data:
        img = imageio.imread(file)
        img_data = data[file]
        split_img = split_triptych_augmented(img, img_data)
        print(file)

        r = split_img[:, :, 0]
        g = split_img[:, :, 1]
        b = split_img[:, :, 2]

        col_align_img = align_and_combine(
            r, g, b, normalized_cross_correlation_without_normalization
        )
        new_file_name = "colored_" + str(i) + ".png"
        imageio.imsave(new_file_name, col_align_img)

        i += 1

    # part 2.3 pyramid data


def part3():
    inside_pic = imageio.imread("rubik/indoor.png")
    outside_pic = imageio.imread("rubik/outdoor.png")

    _, ax = plt.subplots(4, 3)

    ax[0][0].imshow(inside_pic[:, :, 0], cmap="gray", vmin=0, vmax=255)
    ax[0][1].imshow(inside_pic[:, :, 1], cmap="gray", vmin=0, vmax=255)
    ax[0][2].imshow(inside_pic[:, :, 2], cmap="gray", vmin=0, vmax=255)

    ax[1][0].imshow(outside_pic[:, :, 0], cmap="gray", vmin=0, vmax=255)
    ax[1][1].imshow(outside_pic[:, :, 1], cmap="gray", vmin=0, vmax=255)
    ax[1][2].imshow(outside_pic[:, :, 2], cmap="gray", vmin=0, vmax=255)

    new_inside_pic = rgb2lab(inside_pic[:, :, :3])
    new_outside_pic = rgb2lab(outside_pic[:, :, :3])

    ax[2][0].imshow(new_inside_pic[:, :, 0], cmap="gray", vmin=0, vmax=100)
    ax[2][1].imshow(new_inside_pic[:, :, 1], cmap="gray", vmin=-128, vmax=127)
    ax[2][2].imshow(new_inside_pic[:, :, 2], cmap="gray", vmin=-128, vmax=127)

    ax[3][0].imshow(new_outside_pic[:, :, 0], cmap="gray", vmin=0, vmax=100)
    ax[3][1].imshow(new_outside_pic[:, :, 1], cmap="gray", vmin=-128, vmax=127)
    ax[3][2].imshow(new_outside_pic[:, :, 2], cmap="gray", vmin=-128, vmax=127)

    plt.show()

    # img1 = imageio.imread('im1.jpg')
    # img2 = imageio.imread('im2.jpg')

    # x = resize(img1, (256, 256))
    # plt.imshow(x[128:160, 128:160])

    # y = resize(img2, (256, 256))
    # plt.imshow(y[128:160, 128:160])

    # new_img1 = x[128:160, 128:160]
    # new_img2 = y[128:160, 128:160]

    # imageio.imsave('im1.jpg', x)
    # imageio.imsave('im2.jpg', y)


def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
