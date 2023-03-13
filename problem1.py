import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import math
from numpy import unravel_index
from pprint import pprint

# print("inside project 1 file")
cap = cv2.VideoCapture('project2.avi')
if (cap.isOpened()==False):
    print("error on opening ")
def hough(frame, edge, RhoResolution, ThetaResolution):
    # print("inside hough")
    hieght, width = edge.shape
    edge_height_half = hieght/2
    edge_width_half = width/2
    thetas = np.linspace(-90, 90, math.ceil(180/ThetaResolution)+1)
    diag_dist = math.sqrt((hieght-1)**2+(width-1)**2)
    q = math.ceil(diag_dist/RhoResolution)
    nrho = 2*q+1
    rhos  = np.linspace(-q*RhoResolution, q*RhoResolution, nrho)
    hough_space = np.zeros([np.size(rhos), np.size(thetas)])
    theta_rads = thetas*math.pi/180
    cost = np.cos(theta_rads)
    sint = np.sin(theta_rads)
    accumulator = np.zeros((len(rhos), len(rhos)))

    for y in range(hieght):
        for x in range(width):
            if(edge[y, x]):
                edge_point = [y - edge_height_half, x - edge_width_half]
                ys, xs = [], []
                # r = np.round(r/RhoResolution)+q+1
                for thetaIdx in range(np.size(theta_rads)):
                    rho = (edge_point[1])*cost[thetaIdx]+(edge_point[0])*sint[thetaIdx]
                    theta = thetas[thetaIdx]
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    #this is the hough space
                    accumulator[rho_idx][thetaIdx] += 1
                    ys.append(rho)
                    xs.append(theta)
                    # hough_space[int(rhos[thetaIdx]), thetaIdx]=hough_space[int(rhos[thetaIdx]), thetaIdx]+0.25
    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # accumulator = cv2.erode(accumulator, kernel, iterations=1) 
    # accumulator = cv2.dilate(accumulator, kernel, iterations=1)

    cv2.imshow('accumulator', accumulator)

    for y in range(0, accumulator.shape[0], 1):
        for x in range(0, accumulator.shape[1], 1):
            if accumulator[y][x] > 90:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + edge_width_half
                y0 = (b * rho) + edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                print(x1)
                print(x2)
                print(y1)
                print(y2)
                frame = cv2.line(frame, [x1, y1], [x2, y2], [0,0,255], 1)
    return theta, rho
while (cap.isOpened()):
    # Capture frame-by-frame
    t_lower = 100  # Lower Threshold
    t_upper = 200
    L2Gradient = True # Boolean
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_mask_limit = np.array([63, 6, 210])
    upper_mask_limit = np.array([152, 134, 255])
    mask = cv2.inRange(hsv, lower_mask_limit, upper_mask_limit)
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(mask, (17, 17), 0)
    edge = cv2.Canny(blurred, t_lower, t_upper, L2gradient = L2Gradient )
    theta, rho = hough(frame, edge, 3,3)
    # print(unravel_index(hough_space.argmax(), hough_space.shape))
    # print(hough_space.max())
    # print(hough_space.shape)
    # pprint(hough_space)
    if ret == True:
    
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        cv2.imshow("mask", mask)
        cv2.imshow('edge', edge)
        # cv2.imshow('hough_space', hough_space)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Break the loop

    else: 
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
if __name__ == '__main__':
    main(sys.argv)