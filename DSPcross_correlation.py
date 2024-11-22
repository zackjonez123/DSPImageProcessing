"""_summary_

Returns:
    _type_: _description_
"""

import numpy as n
import cv2
import os
import edges

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

def stats(corr_type):
    #print(conv1)
    max1 = n.max(corr_type)
    stats = max1
    return stats

def evaluation(in_path, test_path):
    dir_list1 = os.listdir(in_path)
    dir_list2 = os.listdir(test_path)
    inp_maxs = []

    # Run stats on input images
    for j in range(len(dir_list1)):
        for i in range(len(dir_list2)):
            in_img = cv2.imread(in_path+"\\"+dir_list1[j])
            test_img = cv2.imread(in_path+"\\"+dir_list2[i])
            corr_type = cross_corr(in_img, test_img)
            x = stats(corr_type)
            inp_maxs.append(x) # Appends max correlation
    
    in_stats = n.max(inp_maxs) 
    test_stats = n.min(inp_maxs) # Takes the minimum of the max correlation results
    results = [in_stats, test_stats]
    return results

def evaluation_filter(in_path, test_path):
    dir_list1 = os.listdir(in_path)
    dir_list2 = os.listdir(test_path)
    inp_maxs = []

    # Run stats on input images
    for j in range(len(dir_list1)):
        for i in range(len(dir_list2)):
            in_img = cv2.imread(in_path+"\\"+dir_list1[j])
            test_img = cv2.imread(in_path+"\\"+dir_list2[i])
            corr_type = filter_cross_corr(in_img, test_img)
            x = stats(corr_type)
            inp_maxs.append(x) # Appends max correlation
    
    in_stats = n.max(inp_maxs) 
    test_stats = n.min(inp_maxs) # Takes the minimum of the max correlation results
    results = [in_stats, test_stats]
    return results
    
def main():
    # For cross-correlating one full-sized image on top of another full-sized image
    # "in_path" is for images of the expected (friendly) person in the doorway
    # "test_path" is for images of different people in the doorway

    # in_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zackv2"
    # test_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\test"

    # Original vs Original correlation
    in_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\DSPImageProcessing\\lena.jpg"
    test_path = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\DSPImageProcessing\\lena_crop.jpg"
    in_image = cv2.imread(in_path)
    gray_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    corr_type = cross_corr(gray_image, gray_image)
    corr_max = n.max(corr_type)
    # print(corr_max)
    # # Original vs cropped Original correlation
    crop_img = cv2.imread(test_path)
    kernel = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    fil_corr = filter_cross_corr(gray_image, kernel)
    fil_max = n.max(fil_corr)
    # print(fil_max)
    # Filtered + Edge Detection
    # print(n.shape(gray_image))
    # print(n.shape(kernel))
    in_edge = edges.canny(gray_image)
    fil_edge = edges.canny(kernel)
    fil_edge_corr = filter_cross_corr(in_edge, fil_edge)
    fil_edge_max = n.max(fil_edge_corr)
    print(fil_edge_max)
    # # "Friendly" occupant vs. "Friendly" occupant
    # y = evaluation(in_path, in_path)
    # print("Friendly Occupant vs. Friendly occupant")
    # print("The maximum cross-correlation = ", y[0], "The minimum cross-correlation = ", y[1])

    # "Friendly" occupant vs. "Hostile" occupant
    # x = evaluation(in_path, test_path)
    # print("Friendly Occupant vs. Hostile occupant")
    # print("The maximum cross-correlation = ", x[0], "The minimum cross-correlation = ", x[1])

    # Repeat for filter image
    


    # path1 = "C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\zackv2\\zackdark11.jpg"
    # img = cv2.imread(path1)
    # result = cross_corr(img, img)
    # print(result)

if __name__ == '__main__':
    main()