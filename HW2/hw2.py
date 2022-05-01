import cv2
import numpy as np
import random
import math
import sys
import os
from tqdm import tqdm

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def SIFT(img):
    SIFT_detector = cv2.SIFT_create()
    keypoints, descriptors = SIFT_detector.detectAndCompute(img, None)

    return keypoints, descriptors

class DMatch():
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

class Feat_Matcher():
    def __init__(self, threshold, k):
        self.threshold = threshold
        self.k = k

    def knn_match(self, img1_des, img2_des):
        match_list = []
        for i in tqdm(range(len(img1_des))):
            distance_list = []
            idx_list = []
            for j in range(len(img2_des)):
                distance = np.sqrt(np.sum(np.square(img1_des[i] - img2_des[j])))
                distance_list.append(distance)
                idx_list.append(j)

            idx_list = np.argsort(np.asarray(distance_list))
            distance_list = np.sort(np.asarray(distance_list))
            
            matches = []
            for j in range(self.k):
                match = DMatch(i, idx_list[j], distance_list[j])
                matches.append(match)
            match_list.append(matches)
        
        return match_list

    def matcher(self, img1_kp, img1_des, img2_kp, img2_des):
        match_list = self.knn_match(img1_des, img2_des)

        # ratio test
        good_points = []
        for m, n in match_list:
            if m.distance < self.threshold*n.distance:
                good_points.append([m])
        
        matches = []
        for pair in good_points:
            matches.append(img1_kp[pair[0].queryIdx].pt + img2_kp[pair[0].trainIdx].pt)
        matches = np.array(matches)

        return matches

class RANSAC():
    def __init__(self, threshold, iters):
        self.threshold = threshold
        self.iters = iters

    def get_samples(self, matches):
        # get 4 pairs of samples to build H
        samples = random.sample(range(len(matches)), 4)
        points = [matches[i] for i in samples]

        return np.array(points)

    def get_homography(self, samples):
        A = []
        for sample in samples:
            p1 = np.append(sample[0:2], 1)
            p2 = np.append(sample[2:4], 1)
            A.append([p1[0], p1[1], p1[2], 0, 0, 0, -p1[0]*p2[0], -p1[1]*p2[0], -p1[2]*p2[0]])
            A.append([0, 0, 0, p1[0], p1[1], p1[2], -p1[0]*p2[1], -p1[1]*p2[1], -p1[2]*p2[1]])
        A = np.array(A)

        # svd decomposition
        u, s, vh = np.linalg.svd(A)
        H = (vh[-1] / vh[2, 2]).reshape(3, 3)

        return H
        
    def get_loss(self, matches, H):
        # get err2_estimate
        err2_estimate = []
        for match in matches:
            temp = np.append(match[0:2], 1)
            temp = np.dot(H, temp.T)
            err2_estimate.append((temp/temp[2])[0:2])

        # get err2
        err2 = matches[:, 2:4]

        # get loss
        loss = np.linalg.norm(err2 - err2_estimate, axis=1) ** 2

        return loss

    def ransac(self, matches):
        best_inliers_num = 0
        H_best = np.zeros((3,3))

        for i in tqdm(range(self.iters)):
            # build homography matrix
            samples = self.get_samples(matches)
            H = self.get_homography(samples)
        
            if np.linalg.matrix_rank(H) < 3:
                continue
            
            # compute loss
            loss = self.get_loss(matches, H)
            inliers = matches[np.where(loss < self.threshold)[0]]

            if len(inliers) > best_inliers_num:
                best_inliers_num = len(inliers)
                H_best = H.copy()

        return H_best

class Stitcher():
    def __init__(self, mask_size):
        self.mask_size = mask_size
    
    def create_mask(self, barrier_x, barrier_y, size, side):
        offset = int(self.mask_size / 2)
        barrier = barrier_x + offset
        barrier_h = size[1] - barrier_y
        mask = np.zeros((size[1], size[0]))

        if side == "left":
            mask[barrier_y:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (barrier_h, 1))
            mask[:, :barrier - offset] = 1
            # this part can be adjust
            mask[:barrier_y, :] = 1
        else:
            mask[barrier_y:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (barrier_h, 1))
            mask[:, barrier + offset:] = 1
            # # this part can be adjust
            # mask[:barrier_y, :] = 1
        
        return cv2.merge([mask, mask, mask])

    def stitch_2img(self, img1_rgb, img2_rgb, H, mode):
        h_l = img1_rgb.shape[0]
        w_l = img1_rgb.shape[1]
        h_r = img2_rgb.shape[0]
        w_r = img2_rgb.shape[1]

        # get new corner
        corners_old = [
            [0, 0, 1],
            [w_l-1, 0, 1],
            [w_l-1, h_l-1, 1],
            [0, h_l-1, 1]
        ]
        corners_new = np.array([np.dot(H, corner) for corner in corners_old]).T

        # get translated H
        d_x = min(min(corners_new[0] / corners_new[2]), 0)
        d_y = min(min(corners_new[1] / corners_new[2]), 0)
        A = np.array([[1, 0, -d_x], [0, 1, -d_y], [0, 0, 1]])
        H = np.dot(A, H)

        # warp img1
        size = (round(w_r + abs(d_x)), round(h_r + abs(d_y)))
        warped_l = cv2.warpPerspective(src=img1_rgb, M=H, dsize=size)

        # warp img2
        warped_r = cv2.warpPerspective(src=img2_rgb, M=A, dsize=size)

        # stitching
        stitch_img = np.zeros((size[1], size[0], 3))
        if mode == 0:
            for i in tqdm(range(warped_r.shape[0])):
                for j in range(warped_r.shape[1]):
                    pixel_l = warped_l[i, j, :]
                    pixel_r = warped_r[i, j, :]
                    black = np.zeros(3)

                    if np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                        warped_l[i, j, :] = pixel_r
                    else:
                        warped_l[i, j, :] = pixel_l
            stitch_img = warped_l
        else:
            barrier_x = round(np.dot(A, [0, 0, 1])[0])
            barrier_y = round(np.dot(A, [0, 0, 1])[1])

            mask_l = self.create_mask(barrier_x, barrier_y, size, "left")
            mask_r = self.create_mask(barrier_x, barrier_y, size, "right")
            warped_l = warped_l * mask_l
            warped_r = warped_r * mask_r
            stitch_img = warped_l + warped_r

        return stitch_img


if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)

    # set read path
    img1 = "r56789_v2.jpg"
    img2 = "m10.jpg"
    img_path = "images"

    # read images, img1: left, img2: right
    img1_rgb, img1_gray = read_img(os.path.join(img_path, img1))
    img2_rgb, img2_gray = read_img(os.path.join(img_path, img2))

    # 1. detect keypoints - SIFT
    img1_kp, img1_des = SIFT(img1_gray)
    img2_kp, img2_des = SIFT(img2_gray)
    
    # 2. feature matching - KNN and Lowe's Ratio test
    print("feature matching...")
    matches = Feat_Matcher(threshold=0.75, k=2).matcher(img1_kp, img1_des, img2_kp, img2_des)

    # 3. computing optimal homography matrix - RANSAC
    print("computing best H...")
    H_best = RANSAC(threshold=8, iters=3000).ransac(matches)

    # 4. stitching images
    # mode 0: only stitch, mode 1: stitch with blending
    print("stitching...")
    stitch_img = Stitcher(mask_size=400).stitch_2img(img1_rgb, img2_rgb, H_best, mode=1)

    # show and store result
    creat_im_window("stitch", stitch_img)
    im_show()

    cv2.imwrite("result.jpg", stitch_img)