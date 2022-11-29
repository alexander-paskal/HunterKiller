import cv2 as cv2 #importing the library
import numpy as np
import matplotlib.pyplot as plt

def show_image(name,img): #function for displaying the image
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray,(7,7),0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    # show_image('Canny',img_canny)
    return img_canny

def region_of_interest(image): #function for extracting region of interest
    #bounds in (x,y) format
    # bounds = np.array([[[0,250],[0,200],[150,100],[500,100],[650,200],[650,250]]],dtype=np.int32)
    bounds = np.array([[[96,540],[350,324],[600,324],[864,540]]],dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,0,0])
    masked_image = cv2.bitwise_and(image,mask)
    # show_image('mask',mask) 
    return masked_image


def draw_lines(img,lines): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

def get_coordinates(img,line_parameters): #functions for getting final coordinates
    slope=line_parameters[0]
    intercept = line_parameters[1]
    # y1 =300
    # y2 = 120
    y1=img.shape[0]
    y2 = 0.6*img.shape[0]
    x1= int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,int(y1),x2,int(y2)]

def compute_average_lines(img,lines):
    left_lane_lines=[]
    right_lane_lines=[]
    left_weights=[]
    right_weights=[]
    for points in lines:
        x1,y1,x2,y2 = points[0]
        if x2==x1:
            continue     
        parameters = np.polyfit((x1,x2),(y1,y2),1) #implementing polyfit to identify slope and intercept
        slope,intercept = parameters
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slope <0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)         
        else:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)
    #Computing average slope and intercept
    left_average_line = np.average(left_lane_lines,axis=0)
    right_average_line = np.average(right_lane_lines,axis=0)
    print(left_average_line,right_average_line)
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img,left_average_line)
    right_fit_points = get_coordinates(img,right_average_line) 
    print(left_fit_points,right_fit_points)
    return [[left_fit_points],[right_fit_points]] #returning the final coordinates


def main(path):
    image2 = cv2.imread(path)  # put in cone image
    print(image2.shape)  # prints original image of cone
    # cv2.imshow('input',image2)

    lane_image_2 = np.copy(image2)

    # lane_image_2 = cv2.cvtColor(lane_image_2,cv2.COLOR_BGR2RGB)
    lane_image_2 = cv2.cvtColor(lane_image_2, cv2.COLOR_BGR2HSV)
    # show_image('input',lane_image_2)
    # show_image('hsv_input',lane_image_2)

    # Defining upper and lower boundaries for white color
    lower_white_hls = np.uint8([0, 170, 0])
    upper_white_hls = np.uint8([5, 250, 255])
    # #Using bitwise operators to segment out white colors
    lane_white_mask = cv2.inRange(lane_image_2, lower_white_hls, upper_white_hls)

    kernel = np.ones((15, 15), np.uint8)
    lane_image_2 = cv2.morphologyEx(lane_white_mask, cv2.MORPH_CLOSE, kernel)
    kerenel_dilate = np.ones((7, 7), np.uint8)
    lane_image_3 = cv2.dilate(lane_image_2, kerenel_dilate, iterations=1)

    lane_image_mask = cv2.bitwise_and(lane_image_2, lane_image_2, mask=lane_white_mask)

    contours, heirarchy = cv2.findContours(lane_image_mask, 1, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(image2, cnt, -1, (0, 255, 3), 3)

    return image2
