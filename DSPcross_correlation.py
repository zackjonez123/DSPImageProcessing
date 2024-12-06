"""**** Computes and plots cross-correlation of image sets with different methods ****
    (1) Without Filter 
    (2) With Filter 
    (3) Without Filter
    Also prints the maximum and minimum correlation results for (2) and (3).
    By comparing the maximum and minimum, a threshold for "Friendly" images can be set.
    Minimum --> lower limit, Maximum --> upper limit
"""

import numpy as n
import cv2
import os
import edges

from matplotlib import pyplot as plt

'''
Computes the cross-correlation without Filter image.
Loops through the image, pixel by pixel, multiplies and sums each pixel's RGB values and appends them to an array. 

Params: 
    in_image: image to be cross-correlated
    kernel: Filter image

Returns: 
    corr_arr: array of cross-correlation results for an image
'''
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

'''
Computes the cross-correlation with Filter image.
Loops through the image, pixel by pixel, multiplies and sums each pixel's RGB values and appends them to an array. 

Params: 
    in_image: image to be cross-correlated
    kernel: Filter image

Returns: 
    fil_corr_arr: array of cross-correlation results for an image
'''
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

'''
Computes the cross-correlation without Filter image for each image within an image set.
Determines the maximum correlation result for each image.

Params: 
    path: directory of image set

Returns: 
    maxs: array of maximum values for each image in the set
'''
def original(path):
    dir_list = os.listdir(path)
    maxs = []
    # Run stats on input images
    for i in range(len(dir_list)):
        img = cv2.imread(path+"\\"+dir_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corr_type = cross_corr(gray, gray)
        corr_max = n.max(corr_type) / 1000 # Normalize results
        maxs.append(corr_max)
    return maxs   

'''
Computes the cross-correlation with Filter image for each image within an image set.
Determines the maximum correlation result for each image.

Params: 
    path: directory of image set
    kernel: Filter image

Returns: 
    maxs: array of maximum values for each image in the set
'''
def filtered(path, kernel):
    dir_list = os.listdir(path)
    maxs = []
    # Run stats on input images
    for i in range(len(dir_list)):
        img = cv2.imread(path+"\\"+dir_list[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corr_type = filter_cross_corr(gray, kernel)
        corr_max = n.max(corr_type) / 1000 # Normalize results
        maxs.append(corr_max)
    return maxs  

'''
Computes the cross-correlation with Filter and Edge Detection for each image within an image set.
Both the input image and Filter image are processed with Edge Detection before correlation.
Determines the maximum correlation result for each image.

Params: 
    path: directory of image set
    kernel: Filter image

Returns: 
    maxs: array of maximum values for each image in the set
'''
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

'''
Runs the above functions and plots their results.
Prints the maximum and minimum results for Filter and Filter with Edge Detection.
'''
def main():
    # Paths to images
    in_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOccupied" # Path to "Friendly" image set
    test_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\coleog" # Path to "Hostile" image set
    kpath = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\filter1.jpg" # Path to Filter image
    
    x = original(in_path) # No Filter correlation for "Friendly" images
    fil = cv2.imread(kpath) # Read Filter image
    kernel = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY) # Convert Filter image to grayscale
    y = filtered(in_path, kernel) # Filter correlation for "Friendly" images
    z = edge(in_path, kernel) # Filter with Edge Detection for "Friendly" images
    w = filtered(test_path, kernel) # Filter correlation of "Hostile" images
    v = edge(test_path, kernel) # Edge correlation with Filter of "Hostile" images

    # Create 100 sample points for the 100 images of each case
    N = []
    for i in list(range(1, 101)):
        N.append(i)

    # Plot the maximum correlation values for 100 images
    # Plot "Friendly" image set with no Filter
    p1 = plt.scatter(N, x, c='r')
    plt.title('Friendly: No Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('original.png')

    # Plot "Friendly" image set with Filter
    p2 = plt.scatter(N, y, c='r')
    plt.title('Friendly: With Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('filtered.png')

    # Plot "Friendly" image set with Filter and Edge Detection
    p3 = plt.scatter(N, z, c='r')
    plt.title('Friendly: Filter w/ Edge Detection')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('edge.png')

    # Plot "Hostile" image set with Filter
    p4 = plt.scatter(N, w, c='r')
    plt.title('Hostile: With Filter')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('filtered_test.png')

    # Plot "Hostile" image set with Filter and Edge Detection
    p5 = plt.scatter(N, v, c='r')
    plt.title('Hostile: Filter w/ Edge Detection')
    plt.xlabel('Sample (n)')
    plt.ylabel('Max Correlation')
    plt.savefig('edge_test.png')

    # Setting threshold for Filter
    print("Filter: Friendly max", n.max(y))
    print("Filter: Friendly min", n.min(y))
    print("Filter: Hostile max", n.max(w))
    print("Filter: Hostile min", n.min(w))

    # Setting threshold for Filter with Edge Detection
    print("Filter w/ Edge: Friendly max", n.max(z))
    print("Filter w/ Edge: Friendly min", n.min(z))
    print("Filter w/ Edge: Hostile max", n.max(v))
    print("Filter w/ Edge: Hostile min", n.min(v))
    
if __name__ == '__main__':
    main()