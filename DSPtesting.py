
import os
import cv2
import DSPcross_correlation


def Filter_Test(path, kernel, thresh):
    f_count = 0
    h_count = 0
    res = []
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        res = DSPcross_correlation.filtered(path, kernel)
        if res[i] > thresh[0] and res[i] < thresh[1]:
            f_count += 1
        else:
            h_count += 1
    fvh = [f_count, h_count] # Friendly vs Hostile
    return fvh

def Edge_Test(path, kernel, thresh):
    f_count = 0
    h_count = 0
    res = []
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        res = DSPcross_correlation.edge(path, kernel)
        if res[i] > thresh[0] and res[i] < thresh[1]:
            f_count += 1
        else:
            h_count += 1
    fvh = [f_count, h_count] # Friendly vs Hostile
    return fvh

def main():
    print("***Confusion Matrix***")
    test_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\testcrop\\coleog'
    in_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\classes\\croppedOccupied'
    data_path = 'C:\\Users\\kelly\\Desktop\\IDEs and Sims\\IntruderDef\\pics\\filters\\filter1.jpg'
    fil = cv2.imread(data_path) # Read filter image
    kernel = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY) # Convert filter image to grayscale
    fil_thresh = [0.20, 0.24]
    edge_thresh = [24, 27]
    print("*Friendly case*")
    res1 = Filter_Test(in_path, kernel, fil_thresh)
    print("W/ Filter Friendly ID: Expected = 100, Actual = ", res1[0])
    print("W/ Filter Hostile ID: Expected = 0, Actual = ", res1[1])
    res2 = Edge_Test(in_path, kernel, edge_thresh)
    print("Filter w/ Edge: Friendly ID: Expected = 100, Actual = ", res2[0])
    print("Filter w/ Edge: Hostile ID: Expected = 0, Actual = ", res2[1])

    print("*Hostile case*")
    res3 = Filter_Test(test_path, kernel, fil_thresh)
    print("W/ Filter Friendly ID: Expected = 0, Actual = ", res3[0])
    print("W/ Filter Hostile ID: Expected = 100, Actual = ", res3[1])
    res4 = Edge_Test(in_path, kernel, edge_thresh)
    print("Filter w/ Edge: Friendly ID: Expected = 0, Actual = ", res4[0])
    print("Filter w/ Edge: Hostile ID: Expected = 100, Actual = ", res4[1])

if __name__ == '__main__':
    main()