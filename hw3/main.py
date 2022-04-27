"""
main.py for HW3.

feel free to include libraries needed
"""
import numpy as np
from matplotlib import pyplot as plt
from common import read_img, save_img
import cv2
import pickle
import random


def smallestN_indices(a, N):
    idx = a.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, a.shape)).T


def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


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
    p1_transform = np.load('p1/transform.npy')

    N = p1_transform.shape[0]
    matrix_A = np.zeros((2 * N, 6))
    matrix_B = np.zeros((2 * N, 1))

    A_vals = []
    B_vals = []

    for i in range(N):
        row = p1_transform[i]
        row1 = np.array([row[0], row[1], 0 , 0, 1, 0])
        row2 = np.array([0, 0, row[0], row[1], 0, 1])
        A_vals.append(row1)
        A_vals.append(row2)
        
        B_vals.append(row[2])
        B_vals.append(row[3])

    for i in range(2 * N):
        matrix_A[i] = A_vals[i]
        matrix_B[i] = B_vals[i]

    ans = np.linalg.lstsq(matrix_A, matrix_B, rcond=None)[0]
    print(ans)

    S_matrix = ans[0:4].reshape(2, 2)
    print(S_matrix)

    T_matrix = ans[4:].reshape(1, 2).T
    T_matrix

    estimates = np.zeros((N, 2))

    for i in range(N):
        original_x = p1_transform[i][0]
        original_y = p1_transform[i][1]
        
        originals = np.zeros((2, 1))
        originals[0][0] = original_x
        originals[1][0] = original_y
        
        estimate_x = (np.dot(S_matrix, originals) + T_matrix)
        estimate_y = (np.dot(S_matrix, originals) + T_matrix)
        
        estimates[i][0] = estimate_x[0]
        estimates[i][1] = estimate_y[1]

    plt.scatter(p1_transform[:, 0], p1_transform[:, 1], c='purple', label='Original')
    plt.scatter(p1_transform[:, 2], p1_transform[:, 3], c='green', label='Original Estimates')
    plt.scatter(estimates[:, 0], estimates[:, 1], c='red', label='Actual Estimates')
    plt.legend()

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

    pts_list = []
    for dist in res:
        (x1, y1) = left_kp[dist[0]].pt
        (x2, y2) = right_kp[dist[1]].pt
        pts_list.append([x1, y1, x2, y2])
    
    # 2.5 draw matches & run ransac
    matches = []
    for r in res:
        match = cv2.DMatch(r[0], r[1], distance[r[0], r[1]])
        matches.append(match)
        
    # 2.5 - draw matches
    matched_img = cv2.drawMatches(left_img,left_kp,right_img,right_kp,matches, None, flags=2)
    matched_img_name = left_img_name[:left_img_name.find("_")] + "_matches.jpg"
    save_img(matched_img, matched_img_name)

    # 2.5 - run ransac
    max_inliers = []
    final_H = None
    little_h = None
    random_pts = None
    inliers = []
    d_sum = 0
    d_count = 0

    for i in range(10000):
        d_sum = 0
        d_count = 0

        entry1 = pts_list[random.randrange(0, len(pts_list))]
        entry2 = pts_list[random.randrange(0, len(pts_list))]
        random_pts = np.vstack((entry1, entry2))
        
        entry3 = pts_list[random.randrange(0, len(pts_list))]
        random_pts = np.vstack((random_pts, entry3))
        
        entry4 = pts_list[random.randrange(0, len(pts_list))]
        random_pts = np.vstack((random_pts, entry4))
        
        little_h = fit_homography(random_pts)
        inliers = []
        
        for i in range(len(pts_list)):
            actual = np.array([pts_list[i]])
            d = geometricDistance(actual, little_h)
            d_sum += d
            d_count += 1
            
            if d < 5:
                inliers.append(actual)
                
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            final_H = little_h
        
        if len(max_inliers) > (len(pts_list) * 0.60):
            break

    # 2.6 warp perspective & stitch image
    H = final_H
    d_avg = float(d_sum/d_count)

    print(len(max_inliers))
    print(d_avg)

    edges = np.zeros((4, 2))
    edges[0] = np.array([0, 0]) # 0, 0
    edges[1] = np.array([left_img.shape[1] - 1, 0]) # width, height
    edges[2] = np.array([0, left_img.shape[0] - 1]) # 
    edges[3] = np.array([left_img.shape[1] - 1, left_img.shape[0] - 1])

    left_img_edges_transformed = homography_transform(edges, final_H)

    x_transform = abs(left_img_edges_transformed[:, 0].min())
    y_transform = abs(left_img_edges_transformed[:, 1].min())

    shift_matrix = np.zeros((3, 3))
    shift_matrix[0] = np.array([1, 0, x_transform])
    shift_matrix[1] = np.array([0, 1, y_transform])
    shift_matrix[2] = np.array([0, 0, 1])

    complete_matrix = shift_matrix @ final_H

    canvas_width = int(x_transform + right_img.shape[1])
    canvas_height = int(y_transform + right_img.shape[0])

    left_shift_result = cv2.warpPerspective(left_img, complete_matrix, (canvas_width, canvas_height))

    for i in range(int(y_transform), canvas_height):
        for j in range(int(x_transform), canvas_width):
            if left_shift_result[i, j, :].sum() == 0:
                left_shift_result[i, j, :] = right_img[i - int(y_transform), j - int(x_transform), :]

    final_img_name = left_img_name[:left_img_name.find("_")] + ".jpg"
    save_img(left_shift_result, final_img_name)

if __name__ == "__main__":
    # Problem 1
    p1()

    # Problem 2
    p2('p2/uttower_left.jpg', 'p2/uttower_right.jpg', 'uttower')
    p2('p2/bbb_left.jpg', 'p2/bbb_right.jpg', 'bbb')

    # Problem 3
    # add your code for implementing Problem 3
    #
    # Hint:
    # you can reuse functions in Problem 2
