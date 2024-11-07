"""_summary_

Returns:
    _type_: _description_
"""

import numpy as n
import cv2
import os

def cross_corr(in_image, kernel):
    # Array of cross-correltation
    corr_arr = []
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
            convolved = n.sum(n.multiply(splice, kernel))
            norm_convolved = convolved / 1000 # Normalize results 
            corr_arr.append(norm_convolved)

    return corr_arr
    
def stats(in_image, kernel):
    corr1 = cross_corr(in_image, kernel)
    #print(conv1)
    max1 = n.max(corr1)
    min1 = n.min(corr1)
    #print("Max value of"+descript+"is: ", max1)
    #print("Min value of"+descript+"is: ", min1)
    average = n.sum(corr1) / len(corr1)
    #print("Average value of"+descript+"is: ", average)
    
    stats = [max1, min1, average]
    return stats

def evaluation(in_path, test_path, kernel):
    dir_list1 = os.listdir(in_path)
    in_avmax = []
    in_avmin = [] 
    in_avavg = [] 
    inp = []
    dir_list2 = os.listdir(test_path)
    test_avmax = [] 
    test_avmin = [] 
    test_avavg = []
    test = [] 

    # Run stats on test images
    for i in range(len(dir_list2)):
        test_img = cv2.imread(test_path+"\\"+dir_list2[i])
        t = stats(test_img, kernel)
        # test_avmax.append(t[0]) # Appends max correlation
        # test_avmin.append(t[1]) # Appends min correlation
        # test_avavg.append(t[2]) # Appends average correlation
        #test.append(t[3])
        test.append(t[0])

    # Run stats on input images
    for j in range(len(dir_list1)):
        in_img = cv2.imread(in_path+"\\"+dir_list1[j])
        x = stats(in_img, kernel)
        # in_avmax.append(x[0])
        # in_avmin.append(x[1])
        # in_avavg.append(x[2])
        inp.append(x[0])
        #inp.append(x[4])

    # tavg = n.sum(test_avavg) / len(test_avavg)
    # tavgmax = n.sum(test_avmax) / len(test_avmax)
    # tavgmin = n.sum(test_avmin) / len(test_avmin)
    
    test_stats = n.max(test)

    # inavg = n.sum(in_avavg) / len(in_avavg)
    # inavgmax = n.sum(in_avmax) / len(in_avmax)
    # inavgmin = n.sum(in_avmin) / len(in_avmin)
    in_stats = n.min(inp)

    print("Test Image stats are: ", test_stats)
    print("Zack Image stats are: ", in_stats)

def decision(in_path, kernel, threshold):
    img_read = cv2.imread(in_path)
    y = stats(img_read, kernel)
    if y[0] > threshold:
        return False # Friendly
    else:
        return True # Hostile