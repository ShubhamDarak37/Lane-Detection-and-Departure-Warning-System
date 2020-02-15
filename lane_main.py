# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:18:56 2020

@author: shubh
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from calibration import load_calibration
from copy import copy


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:04:31 2020

@author: shubh
"""



class Lane():
    def __init__(self):
        # was the line detected in the last frame or not
        self.detected = False
        #x values for detected line pixels
        self.cur_fitx = None
        #y values for detected line pixels
        self.cur_fity = None
        # x values of the last N fits of the line
        self.prev_fitx = []
        #polynomial coefficients for the most recent fit
        self.current_poly = [np.array([False])]
        #best polynomial coefficients for the last iteration
        self.prev_poly = [np.array([False])]

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly




left_lane = Lane()
right_lane = Lane()
frame_width = 1280
frame_height = 720

LANEWIDTH = 3.7  # highway lane width in US: 3.7 meters
input_scale = 1
output_frame_scale = 4
N = 4 # buffer previous N lines


x = [550,820,1020,250]
y = [550,550,719, 719]
X = [290, 990, 990, 290]
Y = [719, 719, 0, 0]


pts=np.array([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]])
pts.reshape((-1,1,2))

src = np.float32([[x[0], y[0]], [x[1], y[1]],[x[2], y[2]], [x[3], y[3]]]) 
dst = np.float32([[X[0], Y[0]], [X[1], Y[1]],[X[2], Y[2]], [X[3], Y[3]]]) 

s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)



def warper(img, M):

    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def threshold_col_channel(channel, thresh):
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ threshold according to the direction of the gradient
    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    plt.imshow(absgraddir)
    plt.imshow(gray)

    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def find_edges(img, s_thresh=s_thresh, sx_thresh=sx_thresh, dir_thresh=dir_thresh):

    img = np.copy(img)
    # Convert to HSV color space and threshold the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)

    # Sobel x
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # mag_binary = mag_thresh(img, sobel_kernel=3, thresh=m_thresh)
    # # gradient direction
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    #
    # # output mask
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(( (sxbinary == 1) & (dir_binary==1) ) | ( (s_binary == 1) & (dir_binary==1) ))] = 1

    # add more weights for the s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[( (sxbinary == 1) & (s_binary==1) )] = 2

    ave_binary = (combined_binary + c_bi)

    return ave_binary

input_scale =1



def detector(binary_sub, ploty, visualization):

    left_fit, right_fit,check = full_search(binary_sub, visualization=visualization)
    if check == False:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.detected = False
        right_lane.detected = False
        return False
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    std_value = np.std(right_fitx - left_fitx)
    std_value
    if std_value < (85 ):
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        left_lane.detected = True
        right_lane.detected = True
    else:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        if len(left_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
        else:
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        left_lane.detected = False
        right_lane.detected = False
    return True

visualization=False
def full_search(binary_warped, visualization):

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = np.floor(100/input_scale)
    # Set minimum number of pixels found to recenter window
    minpix = np.floor(50/input_scale)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    count=0;
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if visualization:
            cv2.rectangle(out_img,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2)
            cv2.rectangle(out_img,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        if len(good_left_inds) == 0 or len(good_right_inds) == 0:
            count=count+1
       
        
    
    # Concatenate the arrays of indices
     
    
    # Extract left and right line pixel positions
    
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if count >= 6:
        return False,False,False
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Visualization

    # Generate x and y values for plotting
    if visualization:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.subplot(1,2,1)
        plt.imshow(out_img)
        # plt.imshow(binary_warped)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))
        plt.show()

    return left_fit, right_fit,True





def measure_lane_curvature(ploty, leftx, rightx, visualization=True):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

     # choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/(frame_height/input_scale) # meters per pixel in y dimension
    xm_per_pix = LANEWIDTH/(700/input_scale) # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    if leftx[0] - leftx[-1] > 50/input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 50/input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad+right_curverad)/2.0, curve_direction


def off_center(left, mid, right):
    """

    :param left: left lane position
    :param mid:  car position
    :param right: right lane position
    :return: True or False, indicator of off center driving
    """
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * LANEWIDTH - LANEWIDTH /2.0
    else:       # driving left off
        offset = LANEWIDTH /2.0 - b / width * LANEWIDTH

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):

    # Create an image to draw the lines on
    height = undist.shape[0]
    width = undist.shape[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, 715, bottom_r)

    return offcenter, pts

def tracker(binary_sub, ploty, visualization):

    left_fit, right_fit = window_search(left_lane.prev_poly, right_lane.prev_poly, binary_sub, margin=100/input_scale, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.detected = True
        right_lane.detected = True
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        # global tt
        # tt = tt + 1
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.cur_fitx = left_lane.prev_fitx[-1]
        right_lane.cur_fitx = right_lane.prev_fitx[-1]



def window_search(left_fit, right_fit, binary_warped, margin=100, visualization=True):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's easier to find line pixels with windows search
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_lane.prev_poly[0]*(nonzeroy**2) + left_lane.prev_poly[1]*nonzeroy + left_lane.prev_poly[2] - margin)) & (nonzerox < (left_lane.prev_poly[0]*(nonzeroy**2) + left_lane.prev_poly[1]*nonzeroy + left_lane.prev_poly[2] + margin)))
    right_lane_inds = ((nonzerox > (right_lane.prev_poly[0]*(nonzeroy**2) + right_lane.prev_poly[1]*nonzeroy + right_lane.prev_poly[2] - margin)) & (nonzerox < (right_lane.prev_poly[0]*(nonzeroy**2) + right_lane.prev_poly[1]*nonzeroy + right_lane.prev_poly[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # And you're done! But let's visualize the result here as well
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = out_img.astype('uint8')
        window_img = np.zeros_like(out_img)
        plt.imshow(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))

        plt.show()

    return left_fit, right_fit
cap = cv2.VideoCapture("examples/project_new.mp4")
start_frame_number = 50
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
cap.set(3,1280)
cap.set(4,720)
while(True):
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1280,720))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.polylines(frame,[pts],True,(0,255,255),3)
    img_binary = find_edges(frame)
    #plt.imshow(img_binary)
    wrap=warper(img_binary,M)
    binary_sub = np.zeros_like(wrap)
    binary_sub[:, int(150/input_scale):int(-80/input_scale)]  = wrap[:, int(150/input_scale):int(-80/input_scale)]
    #plt.imshow(binary_sub)
    cv2.imshow('output',binary_sub)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
while(True):
    ret, frame = cap.read()
    frame=cv2.resize(frame,(1280,720))
    #cv2.polylines(frame,[pts],True,(0,255,255),3)
    img_binary = find_edges(frame)
    #cv2.imshow('img_binary',img_binary)
    wrap=warper(img_binary,M)
    binary_sub = np.zeros_like(wrap)
    binary_sub[:, int(150/input_scale):int(-80/input_scale)]  = wrap[:, int(150/input_scale):int(-80/input_scale)]
    ploty = np.linspace(0, wrap.shape[0]-1, wrap.shape[0])
    if left_lane.detected:  # start tracker
        tracker(binary_sub, ploty, visualization)
    else:  # start detector
        check=detector(binary_sub, ploty, visualization)
        
    whole_frame = frame
    font = cv2.FONT_HERSHEY_SIMPLEX

    if check==False:
        cv2.putText(whole_frame,"WARNING LANE NOT FOUND", (100,100), font, 1.2, (255,255,0), 3,cv2.LINE_AA)
        cv2.imshow('output',whole_frame)
        left_lane.detected=False
    else:
        left_lane.process(ploty)
        right_lane.process(ploty)
        
        # measure the lane curvature
        curvature, curve_direction = measure_lane_curvature(ploty, left_lane.mean_fitx, right_lane.mean_fitx)
        print(curve_direction)
        print(curvature)
        # compute the car's off-center in meters
        offcenter, pts = compute_car_offcenter(ploty, left_lane.mean_fitx, right_lane.mean_fitx, frame)
        print(offcenter)
        if offcenter >= 0:
            offset= abs(offcenter)
            #offset = abs(offset - 0.6)
            direction = 'Right'
        elif offcenter < 0:
            offset = abs(offcenter)
            #offset = abs(offset - 0.6)
            direction = 'Left'
        img=frame
        whole_frame = img
        
        info_lane = "Lane info: {0}".format(curve_direction)
        info_cur = "Curvature {:6.1f} m".format(curvature)
        info_offset = "Off center: {0} {1:3.1f}m".format(direction, offset)
        cv2.putText(whole_frame, info_lane, (100,100), font, 1.2, (255,255,0), 3,cv2.LINE_AA)
        cv2.putText(whole_frame, info_cur, (100,200), font, 1.2, (255,255,0), 3,cv2.LINE_AA)
        cv2.putText(whole_frame, info_offset, (100,300), font, 1.5, (255,255,0), 3,cv2.LINE_AA)
        
        undist_birdview = warper(img,M)
        
        colorr_warp = np.zeros_like(img).astype(np.uint8)
        
        # create a frame to hold every image
        threshold=0.6
        
        if abs(offcenter) > threshold:  # car is offcenter more than 0.6 m
            # Draw Red lane
            cv2.fillPoly(colorr_warp, np.int_([pts]), (255, 0, 0)) # red
            left_lane.detected=False
        else: # Draw Green lane
            cv2.fillPoly(colorr_warp, np.int_([pts]), (0,255, 0))  # green
    
        new = cv2.warpPerspective(colorr_warp, M_inv, (int(frame_width/input_scale), int(frame_height/input_scale)))
        
        output = cv2.addWeighted(img, 1, new, 0.3, 0)
        cv2.imshow('output',output)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()