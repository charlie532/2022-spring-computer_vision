from calendar import c
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

image_row = 0 
image_col = 0

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image

def pseudo_inverse(met_a, met_b):
    return np.dot(np.dot(np.linalg.inv(np.dot(met_a.T, met_a)), met_a.T), met_b)

if __name__ == '__main__':
    img_namelist = ['bunny', 'star', 'venus']

    for k in range(len(img_namelist)):
        # read info
        img1 = read_bmp('./' + img_namelist[k] + '/pic1.bmp')
        img2 = read_bmp('./' + img_namelist[k] + '/pic2.bmp')
        img3 = read_bmp('./' + img_namelist[k] + '/pic3.bmp')
        img4 = read_bmp('./' + img_namelist[k] + '/pic4.bmp')
        img5 = read_bmp('./' + img_namelist[k] + '/pic5.bmp')
        img6 = read_bmp('./' + img_namelist[k] + '/pic6.bmp')

        light_list = []
        with open('./' + img_namelist[k] + '/LightSource.txt', 'r') as light_file:
            for j in light_file.readlines():
                light_pos = j[7:-2].split(',')
                light_pos = [float(k) for k in light_pos]
                light_list.append(light_pos)
        light_list = np.array(light_list)


        # Normal Estimation
        # use { I=L*Kd*N => Kd*N=(LT*L)^-1*LT*I } to find Kd*N (pseudo-inverse)
        # build L
        temp_list = []
        for j in light_list:
            temp_list.append(j / np.linalg.norm(j))
        L = np.array(temp_list)
        N = np.zeros((image_row, image_col, 3))
        
        # find N
        for x in range(image_row):
            for y in range(image_col):
                # build I
                I = np.array([
                    img1[x][y],
                    img2[x][y],
                    img3[x][y],
                    img4[x][y],
                    img5[x][y],
                    img6[x][y],
                ])
                # find Kd * N
                KdN = pseudo_inverse(L, I)
                KdN = KdN.T
                # normalize
                Kd = np.linalg.norm(KdN)
                if Kd != 0:
                    N[x][y] = KdN / Kd

        # visualize N
        normal_visualization(N)


        # Surface Reconstruction
        # method 2. linear algebra (stored M in sparsed way)
        # get object index and valid pixel amount from img1(mask)
        obj_h, obj_w = np.where(img1 != 0)
        obj_pixnum = np.size(obj_h)
        obj_grey = np.zeros((image_row, image_col))
        for i in range(np.size(obj_h)):
            obj_grey[obj_h[i], obj_w[i]] = i

        # build M and v (sparse matrix)
        M = scipy.sparse.lil_matrix((2*obj_pixnum, obj_pixnum))
        V = np.zeros((2*obj_pixnum, 1))

        for i in range(obj_pixnum):
            h = obj_h[i]
            w = obj_w[i]

            # z_(x+1, y) - z(x, y) = -(nx / nz)
            if img1[h, w+1] != 0:
                M[i*2, i] = -1
                M[i*2, obj_grey[h, w+1]] = 1
                V[i*2] = -(N[h, w, 0] / N[h, w, 2])
            elif img1[h, w-1] != 0:
                M[i*2, obj_grey[h, w-1]] = -1
                M[i*2, i] = 1
                V[i*2] = -(N[h, w, 0] / N[h, w, 2])

            # z_(x, y+1) - z(x, y) = -(ny / nz)
            if img1[h+1, w] != 0:
                M[i*2+1, i] = 1
                M[i*2+1, obj_grey[h+1, w]] = -1
                V[i*2+1] = -(N[h, w, 1] / N[h, w, 2])
            elif img1[h-1, w] != 0:
                M[i*2+1, obj_grey[h-1, w]] = 1
                M[i*2+1, i] = -1
                V[i*2+1] = -(N[h, w, 1] / N[h, w, 2])

        # find Z (solve MT*M*Z = MT*V)
        Z = scipy.sparse.linalg.spsolve(M.T@M, M.T@V)

        # build whole graph D
        D = img1.astype('float')
        for i in range(obj_pixnum):
            D[obj_h[i], obj_w[i]] = Z[i]

        # visualize D
        depth_visualization(D)


        # save .bmp file
        save_ply(D, img_namelist[k] + '.ply')
        show_ply(img_namelist[k] + '.ply')

    # showing the windows of all visualization function
    plt.show()