import numpy as np
from scipy import ndimage as im

def g_blur(img):
    #g_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    g_kernel = np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]]) / 159

    return im.convolve(img, g_kernel)

def sobel(img):
    K_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    K_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    sobel_x = im.convolve(img, K_x)
    sobel_y = im.convolve(img, K_y)

    mag =  np.sqrt(sobel_x**2 + sobel_y**2)

    # getting the direction for each pixel in degrees
    sobel_dir = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # if the direction is negative, add 360 degrees to make it positive
    sobel_dir[sobel_dir < 0] += 360

    # round the direction to the nearest 45 degrees
    sobel_dir = np.round(sobel_dir / 45) * 45

    # Make directions s.t. 0 and 180 are the same etc to avoid 8 different directions
    sobel_dir[sobel_dir > 135] -= 180

    return (mag, sobel_dir)

def non_max_suppression(sobel, sobel_dir):
    M, N = sobel.shape
    Z = np.zeros((M,N))

    for i in range(1,M-1):
        for j in range(1,N-1):
            direction = sobel_dir[i, j]
            if direction == 0:
                if sobel[i,j] == max(sobel[i,j], sobel[i,j+1], sobel[i,j-1]):
                    Z[i,j] = sobel[i,j]
            elif direction == 45:
                if sobel[i,j] == max(sobel[i,j], sobel[i+1,j-1], sobel[i-1,j+1]):
                    Z[i,j] = sobel[i,j]
            elif direction == 90:
                if sobel[i,j] == max(sobel[i,j], sobel[i+1,j], sobel[i-1,j]):
                    Z[i,j] = sobel[i,j]
            elif direction == 135:
                if sobel[i,j] == max(sobel[i,j], sobel[i+1,j+1], sobel[i-1,j-1]):
                    Z[i,j] = sobel[i,j]

    return Z

def threshold(img, low_threshold=0.05, highthreshold=0.09):
    high_th = img.max() * highthreshold
    low_th = high_th * low_threshold
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = 25
    strong = 255
    
    strong_i, strong_j = np.where(img >= high_th)
    
    weak_i, weak_j = np.where((img <= high_th) & (img >= low_th))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak=25, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    # check if it has a strong neighbor
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def canny(img):
    # Convert to grayscale according to how humans percieve color
    # img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    if img.ndim == 3:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    img = g_blur(img)

    sobel_im = sobel(img)

    img = non_max_suppression(sobel_im[0], sobel_im[1])

    img = threshold(img,0.05,0.12)[0]

    img = hysteresis(img)

    return img


