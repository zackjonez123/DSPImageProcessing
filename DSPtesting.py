'''
**** Tests for cross-correlation with Filter and Filter with Edge Detection ****
    "Friendy" images are within the threshold (my range of maximum correlation values)
    "Hostile" images are outside of the threshold
'''

import cv2
import DSPcross_correlation

'''
Runs cross-correlation with Filter for a set of images. 
Counts how many are within the threshold.

Params: 
    path: directory to image set
    kernel: Filter image
    thresh: [low limit, high limit] (for maximum correlation value of an entire image)

Returns: 
    fvh: Friendly (within threshold) versus Hostile (outside threshold) 
'''
def Filter_Test(path, kernel, thresh):
    f_count = 0
    h_count = 0
    res = DSPcross_correlation.filtered(path, kernel)
    for i in range(len(res)):
        if res[i] > thresh[0] and res[i] < thresh[1]:
            f_count += 1
        else:
            h_count += 1
    fvh = [f_count, h_count] # Friendly vs Hostile
    return fvh

'''
Runs cross-correlation for Filter with Edge Detection for a set of images. 
Counts how many are within the threshold.

Params: 
    path: directory to image set
    kernel: Filter image
    thresh: [lower limit, upper limit] (for maximum correlation value of an entire image)

Returns: 
    fvh: Friendly (within threshold) versus Hostile (outside threshold) 
'''
def Edge_Test(path, kernel, thresh):
    f_count = 0
    h_count = 0
    res = DSPcross_correlation.edge(path, kernel)
    for i in range(len(res)):
        if res[i] > thresh[0] and res[i] < thresh[1]:
            f_count += 1
        else:
            h_count += 1
    fvh = [f_count, h_count] # Friendly vs Hostile
    return fvh

'''
Runs the above test functions and prints their results.

'''
def main():
    print("***Confusion Matrix***")

    test_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\coleog' # Path to "Hostile" image set
    in_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOccupied' # Path to "Friendly" image set
    data_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\filter1.jpg' # Path to Filter image

    fil = cv2.imread(data_path) # Read filter image
    kernel = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY) # Convert filter image to grayscale
    fil_thresh = [0.20, 0.24] # Threshold with Filter
    edge_thresh = [24, 27] # Threshold with Filter and Edge Detection

    print("*Friendly case*") # 100 images of myself
    res1 = Filter_Test(in_path, kernel, fil_thresh)
    print("W/ Filter Friendly ID: Expected = 100, Actual = ", res1[0])
    print("W/ Filter Hostile ID: Expected = 0, Actual = ", res1[1])
    res2 = Edge_Test(in_path, kernel, edge_thresh)
    print("Filter w/ Edge: Friendly ID: Expected = 100, Actual = ", res2[0])
    print("Filter w/ Edge: Hostile ID: Expected = 0, Actual = ", res2[1])

    print("*Hostile case*") # 100 images of my brother
    res3 = Filter_Test(test_path, kernel, fil_thresh)
    print("W/ Filter Friendly ID: Expected = 0, Actual = ", res3[0])
    print("W/ Filter Hostile ID: Expected = 100, Actual = ", res3[1])
    res4 = Edge_Test(in_path, kernel, edge_thresh)
    print("Filter w/ Edge: Friendly ID: Expected = 0, Actual = ", res4[0])
    print("Filter w/ Edge: Hostile ID: Expected = 100, Actual = ", res4[1])

if __name__ == '__main__':
    main()