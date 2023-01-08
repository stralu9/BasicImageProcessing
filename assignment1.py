# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:01:59 2022

@author: LucaS
"""

import cv2
import numpy as np

def hough_transform(frame, p1, p2, minR, maxR):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows, param1=p1, param2=p2,
               minRadius=minR, maxRadius=maxR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    return frame

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
    flag = True
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            if between(cap, 0, 4000):
                if between(cap, 0, 1000):
                    out_frame = frame
                    out_frame = cv2.putText(out_frame, 'COLOR SCALE', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255,255,255), thickness=2)
                elif between(cap, 1000, 2000):
                    out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2BGR)
                    out_frame = cv2.putText(out_frame, 'GRAY SCALE', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                elif between(cap, 2000, 3000):
                    out_frame = frame
                    out_frame = cv2.putText(out_frame, 'COLOR SCALE', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                elif between(cap, 3000, 4000):
                    out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_GRAY2BGR)
                    out_frame = cv2.putText(out_frame, 'GRAY SCALE', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                out.write(out_frame)
            elif between(cap, 4000, 12000):
                if between(cap, 4000, 4500):
                    frame = cv2.GaussianBlur(frame,(3,3),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=3 and sigma=1', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 4500, 5000):
                    frame = cv2.GaussianBlur(frame,(5,5),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=5 and sigma=1', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 5000, 5500):
                    frame = cv2.GaussianBlur(frame,(7,7),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=7 and sigma=0', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 5500, 6000):
                    frame = cv2.GaussianBlur(frame,(9,9),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=9 and sigma=1', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 6000, 6500):
                    frame = cv2.GaussianBlur(frame,(11,11),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=11 and sigma=1', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 6500, 7000):
                    frame = cv2.GaussianBlur(frame,(7,7),1)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=7 and sigma=3', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 7000, 7500):
                    frame = cv2.GaussianBlur(frame,(7,7),5)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=7 and sigma=5', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame
                elif between(cap, 7500, 8000):
                    frame = cv2.GaussianBlur(frame,(7,7),7)
                    frame = cv2.putText(frame, 'This filter reduces gaussian noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Gaussian blur with ', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'kernel size=7 and sigma=7', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = frame                    
                elif between(cap, 8000, 8500):
                    frame = cv2.bilateralFilter(frame,5,75,75)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=75', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 5x5 (fast)', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 8500, 9000):
                    frame = cv2.bilateralFilter(frame,7,75,75)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=75', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 7x7 (slower)', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 9000, 9500):
                    frame = cv2.bilateralFilter(frame,9,75,75)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=75', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 9x9', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, '(heavy noise filtering)', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)

                    out_frame = frame
                elif between(cap, 9500, 10000):
                    frame = cv2.bilateralFilter(frame,11,75,75)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=75', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 11x11', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, '(heavy noise filtering)', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 10000, 10500):
                    frame = cv2.bilateralFilter(frame,5,150,150)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=150', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 5x5 (fast)', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 10500, 11000):
                    frame = cv2.bilateralFilter(frame,7,150,150)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping  sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=150', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 7x7 (slower)', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 11000, 11500):
                    frame = cv2.bilateralFilter(frame,9,150,150)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping sharp edges', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=150', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 9x9', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, '(heavy noise filtering)', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                elif between(cap, 11500, 12000):
                    frame = cv2.bilateralFilter(frame,11,150,150)
                    frame = cv2.putText(frame, 'This filter reduces noise', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'keeping edges sharp', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, 'Bilateral filter with sigmas=150', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    frame = cv2.putText(frame, 'and kernel size 11x11', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    frame = cv2.putText(frame, '(heavy noise filtering)', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = frame
                out.write(out_frame)
            elif between(cap, 13000,21000):
                if between(cap,13000,15000):
                    frame = cv2.GaussianBlur(frame,(5,5),0)
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    fr = (19, 95, 100)
                    to = (35, 200, 255)
                    mask = cv2.inRange(frame_hsv, fr, to)
                    out_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    out_frame = cv2.putText(out_frame, 'Ball detection', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'in HSV color space', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)

                elif between(cap,15000,17000):
                    frame = cv2.GaussianBlur(frame,(5,5),0)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fr = (220, 70, 55)
                    to = (255, 255, 149)
                    mask = cv2.inRange(frame_rgb, fr, to)
                    out_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    out_frame = cv2.putText(out_frame, 'Ball detection', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'in RGB color space', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                elif between(cap,17000,21000):                         
                    frame = cv2.GaussianBlur(frame,(5,5),0)
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    fr = (15, 95, 100)
                    to = (43, 200, 255)
                    kernel = np.ones((21,21),np.uint8)
                    mask = cv2.inRange(frame_hsv, fr, to)
                    
                    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
                    
                    mask_diff = mask2 - mask
                    mask_diff2 = mask - mask2
                    final = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
                    difference = cv2.cvtColor(mask_diff, cv2.COLOR_GRAY2BGR)
                    difference = cv2.cvtColor(mask_diff2, cv2.COLOR_GRAY2BGR)
                    difference[mask_diff > 0] = (0,255,0)
                    difference[mask_diff2 > 0] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(difference,1,final,1, 0)
                    out_frame = cv2.putText(out_frame, 'Ball detection in HSV space', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with opening and closing', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Added or removed points are green', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                out.write(out_frame)
            elif between(cap, 21000, 26000):
                if between(cap, 21000, 22000):
                    ddepth = cv2.CV_16S
                    frame = cv2.GaussianBlur(frame, (11, 11), 1)  
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3)
                    # Gradient-Y
                    # grad_y = cv.Scharr(gray,ddepth,0,1)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3)    
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)   
                    
                    x_edges = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                    y_edges = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
                    x_edges[abs_grad_x > 50] = (0,0,255)
                    y_edges[abs_grad_y > 50] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(x_edges, 0.5, y_edges, 0.5, 0)
                    out_frame = cv2.putText(out_frame, 'Horizontal edges : green', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Vertical edges : red', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'SOBEL with Gaussian smoothing', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with sigma = 1', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'and sobel kernel size = 3', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)

                elif between(cap, 22000, 23000):
                    ddepth = cv2.CV_16S
                    frame = cv2.GaussianBlur(frame, (11,11), 3)   
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    x_edges = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                    y_edges = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
                    x_edges[abs_grad_x > 50] = (0,0,255)
                    y_edges[abs_grad_y > 50] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(x_edges, 0.5, y_edges, 0.5, 0)
                    out_frame = cv2.putText(out_frame, 'Horizontal edges : green', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Vertical edges : red', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'SOBEL with Gaussian smoothing', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with sigma = 3', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'and sobel kernel size = 3', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                
                elif between(cap, 23000, 24000):
                    ddepth = cv2.CV_16S
                    frame = cv2.GaussianBlur(frame, (11, 11), 5)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    x_edges = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                    y_edges = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
                    x_edges[abs_grad_x > 50] = (0,0,255)
                    y_edges[abs_grad_y > 50] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(x_edges, 0.5, y_edges, 0.5, 0)
                    out_frame = cv2.putText(out_frame, 'Horizontal edges : green', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Vertical edges : red', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'SOBEL with Gaussian smoothing', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with sigma = 5', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)
                    out_frame = cv2.putText(out_frame, 'and sobel kernel size = 3', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)

                elif between(cap, 24000, 25000):
                    ddepth = cv2.CV_16S
                    frame = cv2.GaussianBlur(frame, (11, 11), 3)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=5)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=5)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)    
                    x_edges = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                    y_edges = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
                    x_edges[abs_grad_x > 50] = (0,0,255)
                    y_edges[abs_grad_y > 50] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(x_edges, 0.5, y_edges, 0.5, 0)
                    out_frame = cv2.putText(out_frame, 'Horizontal edges : green', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Vertical edges : red', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'SOBEL with Gaussian smoothing', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with sigma = 3', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'and sobel kernel size = 5', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                elif between(cap, 25000, 26000):
                    ddepth = cv2.CV_16S
                    frame = cv2.GaussianBlur(frame, (11, 11), 3)    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=5)
                    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=5)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    x_edges = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                    y_edges = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)
                    x_edges[abs_grad_x > 50] = (0,0,255)
                    y_edges[abs_grad_y > 50] = (0,255,0)
                    
                    out_frame = cv2.addWeighted(x_edges, 0.5, y_edges, 0.5, 0)
                    out_frame = cv2.putText(out_frame, 'Horizontal edges : green', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'Vertical edges : red', (50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'SOBEL with Gaussian smoothing', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'with sigma = 5', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                    out_frame = cv2.putText(out_frame, 'and sobel kernel size = 5', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                out.write(out_frame)
            elif between(cap,26000,36000):
                if between(cap,26000,28000):
                    out_frame = hough_transform(frame, 60, 20, 40, 100)
                    out_frame = cv2.putText(out_frame, 'Hough circles', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'with param1=60, param2=20 ', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                elif between(cap,28000,30000):
                    out_frame = hough_transform(frame, 80, 20, 40, 100)
                    out_frame = cv2.putText(out_frame, 'Hough circles', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'with param1=80, param2=20', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                elif between(cap,30000,32000):
                    out_frame = hough_transform(frame, 100, 20, 40, 100)
                    out_frame = cv2.putText(out_frame, 'Hough circles', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'with param1=100, param2=20', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, '(best configuration)', (50,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)

                elif between(cap,32000,34000):
                    out_frame = hough_transform(frame, 80, 30, 40, 100)
                    out_frame = cv2.putText(out_frame, 'Hough circles', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'with param1=100, param2=30', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                elif between(cap,34000,36000):
                    out_frame = hough_transform(frame, 80, 40, 40, 100)
                    out_frame = cv2.putText(out_frame, 'Hough circles', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'with param1=100, param2=40', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                
            
            
                out.write(out_frame)
            elif between(cap,36000,41000):
                if between(cap, 36000, 38000):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fr = (220, 70, 55)
                    to = (255, 255, 149)
                    mask = cv2.inRange(frame_rgb, fr, to)
                    i, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
                    x,y,w,h = cv2.boundingRect(cont_sorted[0])
        
                    cv2.rectangle(frame,(x-2,y-2),(x+w+2,y+h+2),(0,0,255),3)
                    out_frame = frame
                    out_frame = cv2.putText(out_frame, 'Draw a rectangle around', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'the ball', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2, bottomLeftOrigin=False)
                
                elif between(cap,38000,41000):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.medianBlur(gray, 9)
                    rows = gray.shape[0]
                    
                    mask = np.zeros((rows, gray.shape[1],1), dtype=np.uint8)
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows, param1=100, param2=30,
                               minRadius=40, maxRadius=100)

                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for (x, y, r) in circles:
                            cv2.circle(mask, (x, y), r, 255, -1)
                            
                    h,w,channels = frame.shape
                    method = cv2.TM_SQDIFF_NORMED
                    sub = mask[y-r:y+r+1, x-r:x+r+1]
                    frame = cv2.copyMakeBorder(frame, int(r)-1, int(r), int(r)-1, int(r), cv2.BORDER_CONSTANT,value=[0,0,0])
                    template = cv2.matchTemplate(mask, sub, method)
                    template = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    res = 255 - template
                    l_map = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                    
                    res = cv2.resize(l_map, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    out_frame = res
                    out_frame = cv2.putText(out_frame, 'Likelihood image to find the most', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2, bottomLeftOrigin=False)
                    out_frame = cv2.putText(out_frame, 'probable position of the ball', (50,1150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2, bottomLeftOrigin=False)
                
                    
                out.write(out_frame)
            
            if between(cap,0,41000):
                pass
                
            else:
                break
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
        
        
    cap2 = cv2.VideoCapture("carteBlanche.mp4") 
    while cap2.isOpened():
        ret, frame = cap2.read()
        
        if ret:
            
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            if between(cap2, 0, 20000):
                if between(cap2, 0, 2000):
                    out_frame = frame
                    if flag:
                        background = frame.copy()                        
                        flag = False
                elif between(cap2, 5000,7000):
                    out_frame = frame 
                elif between(cap2, 2000, 5000):                
                    fr = (0, 109, 0)
                    to = (25, 255, 63)
                    frame = cv2.GaussianBlur(frame,(5,5),0)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
                    mask1 = cv2.inRange(hsv, fr, to)               
                    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations = 5)
                    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations = 5)
                    mask2 = cv2.bitwise_not(mask1)
                    res1 = cv2.bitwise_and(background, background, mask=mask1)
                    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
                    out_frame = cv2.addWeighted(res1, 1, res2, 1, 0)
                    out_frame = cv2.putText(out_frame, 'Ball disappears', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                elif between(cap2, 7000, 10000):
                    earth = cv2.imread("earth.jpg")
                    frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    earth = cv2.resize(earth, (frame_width, frame_height))
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                  
                    fr = (0, 80, 0)
                    to = (25, 255, 100)
                    
                    kernel = np.ones((15,15),np.uint8)
                    mask = cv2.inRange(frame_hsv, fr, to)                        
                    
                    mask = cv2.erode(mask, None, iterations=7)
                    
                    thresh = cv2.dilate(mask, None, iterations=7)
                    
                    counts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                    
                    if len(counts)>0:
                        area = max(counts, key=cv2.contourArea)
                    (x, y, w, h) = cv2.boundingRect(area)
                    size = (h + w)//2
                    if y + size < frame_height and x + size < frame_width:
                        logo = cv2.resize(earth, (size, size))
                        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                        _, logo_mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                        roi = frame[y:y+size, x:x+size]
                        roi[np.where(logo_mask)] = 0
                        roi += logo
                    
                    out_frame = frame
                    out_frame = cv2.putText(out_frame, 'Earth image replaces the orange ball', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                elif between(cap2, 10000, 20100):                
                    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                    fr = (0, 80, 0)
                    to = (25, 255, 100)
                    
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, fr, to)
                    mask = cv2.erode(mask, None, iterations=7)
                    mask = cv2.dilate(mask, None, iterations=7)
                    counts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
                    
                    if len(counts) > 0:
                        c = max(counts, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        if radius > 20:
                            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255), 3)
                          
                    out_frame = frame
                    out_frame = cv2.putText(out_frame, 'Detect and follow an orange ball', (50,1100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 255, 255), thickness=2)
                
                out.write(out_frame)
            else:
                break
    
    out.release()
    cap.release()
    cap2.release
    
    # Closes all the frames
    cv2.destroyAllWindows()
            
if __name__ == '__main__':

    main("input.mp4", "output.mp4")