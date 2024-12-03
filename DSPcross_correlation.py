"""_summary_

Returns:
    _type_: _description_
"""

import numpy as n
import cv2
import os
import edges

from matplotlib import pyplot as plt

def cross_corr(in_image, kernel):
    # Array of cross-correltation
    corr_arr = []

    # To keep the filter from going out of the in_image bounds
    width = in_image.shape[1]
    height = in_image.shape[0]

    for i in range(height):
        for j in range(width):
            corr = n.sum(n.multiply(in_image[i, j], kernel[i, j]))
            norm_corr = corr / 1000 # Normalize results 
            corr_arr.append(norm_corr)

    return corr_arr
    
def filter_cross_corr(in_image, kernel):
    # Array of cross-correltation
    fil_corr_arr = []
    # Size of filter image
    fil_width = kernel.shape[1]
    fil_height = kernel.shape[0]
    # Size of input image
    in_width = in_image.shape[1]
    in_height = in_image.shape[0]

    # To keep the filter from going out of the in_image bounds
    width = in_width - fil_width 
    height = in_height - fil_height 

    for i in range(height):
        for j in range(width):
            splice = in_image[i : i + fil_height, j : j + fil_width] # splicing the input image
            corr = n.sum(n.multiply(splice, kernel))
            norm_corr = corr / 1000 # Normalize results 
            fil_corr_arr.append(norm_corr)

    return fil_corr_arr

def original(path):
    dir_list = os.listdir(path)
    maxs = []
    # Run stats on input images
    for i in range(len(dir_list)):
        img = cv2.imread(path+"\\"+dir_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corr_type = cross_corr(gray, gray)
        corr_max = n.max(corr_type) / 1000
        maxs.append(corr_max)
    return maxs   

def filtered(path, kernel):
    dir_list = os.listdir(path)
    maxs = []
    # Run stats on input images
    for i in range(len(dir_list)):
        img = cv2.imread(path+"\\"+dir_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corr_type = filter_cross_corr(gray, kernel)
        corr_max = n.max(corr_type) / 1000
        maxs.append(corr_max)
    return maxs  

def edge(path, kernel):
    dir_list = os.listdir(path)
    maxs = []
    # Run stats on input images
    for i in range(len(dir_list)):
        img = cv2.imread(path+"\\"+dir_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        in_edge = edges.canny(gray)
        fil_edge = edges.canny(kernel)
        fil_edge_corr = filter_cross_corr(in_edge, fil_edge)
        fil_edge_max = n.max(fil_edge_corr) / 1000
        maxs.append(fil_edge_max)
    return maxs  

def main():

    # Paths to images
    in_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOccupied"
    test_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\coleog"
    kpath = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\filter1.jpg"
    
    x = original(in_path) # Full-size original correlation
    fil = cv2.imread(kpath) # Read filter image
    kernel = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY) # Convert filter image to grayscale
    y = filtered(in_path, kernel) # Filter correlation with original images
    z = edge(in_path, kernel) # Edge correlation with filter and original images
    w = filtered(test_path, kernel) # Filter correlation of test images
    v = edge(test_path, kernel) # Edge correlation with filter and test images

    # Create 100 sample points for the 100 images of each case
    N = []
    for i in list(range(1, 101)):
        N.append(i)

    #plot the maxes of 100 images
    p1 = plt.scatter(N, x, c='r')
    plt.title('Friendly: No Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('original.png')

    p2 = plt.scatter(N, y, c='r')
    plt.title('Friendly: With Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('filtered.png')

    p3 = plt.scatter(N, z, c='r')
    plt.title('Friendly: Filter w/ Edge Detection')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('edge.png')

    p4 = plt.scatter(N, w, c='r')
    plt.title('Hostile: With Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('filtered_test.png')

    p5 = plt.scatter(N, v, c='r')
    plt.title('Hostile: Filter w/ Edge Detection')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('edge_test.png')

    # Setting threshold
    print("Filter: Friendly max", n.max(y))
    print("Filter: Friendly min", n.min(y))
    print("Filter: Hostile max", n.max(w))
    print("Filter: Hostile min", n.min(w))

    print("Filter w/ Edge: Friendly max", n.max(z))
    print("Filter w/ Edge: Friendly min", n.min(z))
    print("Filter w/ Edge: Hostile max", n.max(v))
    print("Filter w/ Edge: Hostile min", n.min(v))
    
if __name__ == '__main__':
    main()