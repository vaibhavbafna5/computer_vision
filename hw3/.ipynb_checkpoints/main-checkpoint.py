"""
main.py for HW3.

feel free to include libraries needed
"""
import numpy as np
from matplotlib import pyplot as plt
from common import read_img, save_img
import cv2
import pickle


def smallestN_indices(a, N):
    idx = a.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, a.shape)).T


def homography_transform(X, H):
    # Perform homography transformation on a set of points X
    # using homography matrix H
    #
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix
    # Output - a set of 2D points in an array with size (N,2)
    height = X.shape[0]
    output = np.zeros((height, 2))
    
    for i in range(height):
        output_row = np.array([0, 0])
        second_term = np.array([X[i][0], X[i][1], 1])
        output_row[0] = np.dot(H, second_term)[0]
        output_row[1] = np.dot(H, second_term)[1]
        output[i] = output_row

    return output


def fit_homography(XY):
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    #
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    N = XY.shape[0]
    height = XY.shape[0] * 2
    P = np.zeros((height, 9))
    
    p_vals = []
    
    for i in range(0, N):
        row = XY[i]
    
        x1 = row[0]
        y1 = row[1]
        x1_prime = row[2]
        y1_prime = row[3]
        
        p_vals.append(np.array([ -1 * x1, -1 * y1, -1, 0, 0, 0, x1 * x1_prime, y1 * x1_prime, x1_prime ]))
        p_vals.append(np.array([ 0, 0, 0, -1 * x1, -1 * y1, -1, x1 * y1_prime, y1 * y1_prime, y1_prime ]))
        
    for j in range(0, 2 * N):
        P[j] = p_vals[j]
        
    u, s, v = np.linalg.svd(P)
    
    H = np.reshape(v[8], (3, 3))
    
    H = (1/H.item(8)) * H
    
    return H


def p1():
    # code for Q1.2.3 - Q1.2.5
    # 1. load points X from p1/transform.npy

    # 2. fit a transformation y=Sx+t

    # 3. transform the points
    
    # 4. plot the original points and transformed points

    # code for Q1.2.6 - Q1.2.8
    case = 8  # you will encounter 8 different transformations
    for i in range(case):
        XY = np.load('p1/points_case_'+str(i)+'.npy')
        # 1. generate your Homography matrix H using X and Y
        #
        #    specifically: fill function fit_homography()
        #    such that H = fit_homography(XY)
        H = fit_homography(XY)
        # 2. Report H in your report
        print("case ", i)
        print(H)
        # 3. Transform the points using H
        #
        #    specifically: fill function homography_transform
        #    such that Y_H = homography_transform(X, H)
        Y_H = homography_transform(XY[:, :2], H)
        # 4. Visualize points as three images in one figure
        # the following code plot figure for you
        plt.scatter(XY[:, 1], XY[:, 0], c="red")  # X
        plt.scatter(XY[:, 3], XY[:, 2], c="green")  # Y
        plt.scatter(Y_H[:, 1], Y_H[:, 0], c="blue")  # Y_hat
        plt.savefig('./case_'+str(i))
        plt.close()


def stitchimage(imgleft, imgright):
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv

    # 2. select paired descriptors

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers

    # 4. warp one image by your transformation
    #    matrix
    #
    #    Hint:
    #    a. you can use function of opencv to warp image
    #    b. Be careful about final image size

    # 5. combine two images, use average of them
    #    in the overlap area

    pass


def p2(p1, p2, savename):
    # read left and right images
    
    # 2.1 display grayscale images
    left_img_name = p1
    right_img_name = p2
    
    left_img = read_img(left_img_name)
    right_img = read_img(right_img_name)
    
    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    
    left_img_gray_name = left_img_name[:-4] + "_gray.jpg"
    right_img_gray_name = right_img_name[:-4] + "_gray.jpg"
    
    save_img(left_img_gray, left_img_gray_name)
    save_img(right_img_gray, right_img_gray_name)
    
    # 2.2 use sift descriptors & display feature points
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    
    left_kp, left_descriptors = sift.detectAndCompute(left_img_gray, None)
    left_kp_img = cv2.drawKeypoints(left_img_gray, left_kp, None)
    
    right_kp, right_descriptors = sift.detectAndCompute(right_img_gray, None)
    right_kp_img = cv2.drawKeypoints(right_img_gray, right_kp, None)
    
    left_kp_img_name = left_img_name[:-4] + "_descriptor.jpg"
    right_kp_img_name = right_img_name[:-4] + "_descriptor.jpg"
    
    save_img(left_kp_img, left_kp_img_name)
    save_img(right_kp_img, right_kp_img_name)
    
    # 2.3 calculate distances
    des1 = left_descriptors
    des2 = right_descriptors
    
    distance = np.sqrt(np.sum((des1[:, np.newaxis, :] - des2[np.newaxis, :, :]) ** 2, axis=-1))
    pickle.dump(distance, open('distance.pkl', 'wb'))
    
    # 2.4 select putative matches
    res = smallestN_indices(distance, 250)
    
    # 2.5 draw matches & run ransac
    matches = []
    for r in res:
        match = cv2.DMatch(r[0], r[1], distance[r[0], r[1]])
        matches.append(match)
        
    # 2.5 - draw matches
    matched_img = cv2.drawMatches(left_img,left_kp,right_img,right_kp,matches, None, flags=2)
    matched_img_name = left_img_name[:left_img_name.find("_")] + "_matches.jpg"
    save_img(matched_img, matched_img_name)

    
    
    
    
    
    # stitch image
#     output = stitchimage(imgleft, imgright)
#     # save stitched image
#     save_img(output, './{}.jpg'.format(savename))


if __name__ == "__main__":
    # Problem 1
#     p1()

    # Problem 2
    p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
#     p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

    # Problem 3
    # add your code for implementing Problem 3
    #
    # Hint:
    # you can reuse functions in Problem 2
