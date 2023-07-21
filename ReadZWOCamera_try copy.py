import zwoasi as asi
import os
import cv2
import numpy as np
import time
import imutils
from astropy.io import fits
import matplotlib as plt


#Grab camera frames from ZWO cameras
#PMH 6/29/23
#Following example from https://rk.edu.pl/en/scripting-machine-vision-and-astronomical-cameras-python/
#More code also at: https://github.com/python-zwoasi/python-zwoasi/blob/master/zwoasi/examples/zwoasi_demo.py

#Below didn't seem to work.  Eventaully, this is the write way to setup installation of the .dylib file
#env_filename = os.getenv('ZWO_ASI_LIB')
#print (env_filename)

path = '/Users/piramonkumnurdmanee/Documents/lamat-python/venv/lib/python3.10/site-packages/libASICamera2.dylib'		#hardcoded path. Needs to be adjusted on each computer 

asi.init(path)   #The first time this is run you will need to go to your System Preferences under "Security and Privacy" and click "Open Anyway"

cap = cv2.VideoCapture()

num_cameras = asi.get_num_cameras()
if num_cameras == 0:
    raise ValueError('No cameras found')

camera_id = 0  # use first camera from list
cameras_found = asi.list_cameras()
print(cameras_found)

camera = asi.Camera(camera_id)
camera_info = camera.get_camera_property()
print(camera_info)

# Get all of the camera controls
print('')
print('Camera controls:')
controls = camera.get_controls()
for cn in sorted(controls.keys()):
    print('    %s:' % cn)
    for k in sorted(controls[cn].keys()):
        print('        %s: %s' % (k, repr(controls[cn][k])))

# Use minimum USB bandwidth permitted
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue'])

# Set some sensible defaults. They will need adjusting depending upon
# the sensitivity, lens and lighting conditions used.
camera.disable_dark_subtract()

camera.set_control_value(asi.ASI_GAIN, 100)
camera.set_control_value(asi.ASI_EXPOSURE, 1300000) # microseconds   This saturated the images
camera.set_control_value(asi.ASI_EXPOSURE, 8000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0)


#Documentation says the ASI-290mm has 12 bit depth.  But the SDK has no ASI_IMG_RAW12.  Maybe 16 bit is how you get it?




# Initialize the camera
#camera = ASI_Camera(0)  # Use the correct camera index (e.g., 0 for the first camera)
#camera.open()
"""ret, frame = camera.capture()
if not ret:
    print("Failed to read a frame from the camera.")
    exit()"""
"""
# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the correct camera index (e.g., 0 for the first camera)

# Define the desired crop dimensions (in pixels)
crop_x = 100  # Crop starting x-coordinate
crop_y = 100  # Crop starting y-coordinate
crop_width = 400  # Crop width
crop_height = 300  # Crop height

# This is to find the centroid of a blob
while True:
    # Capture a frame from the camera
    ret, frame = camera.VideoCapture(0)

    # Preprocess the frame (if required)
    # Apply any necessary image processing techniques such as resizing, blurring, or color conversion

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming the blob of interest is the largest)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the largest contour
        M = cv2.moments(largest_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # Display the centroid on the frame
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)

    # Display the video frame with the centroid
    cv2.imshow("Video Stream", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the display window
camera.release()
cv2.destroyAllWindows()

"""





#Create an endless loop to show images from camera
resultname = 'Images'
#setup display of camera results
displaycoloffset = 0000
cv2.namedWindow(resultname)        # Create a window to show captured images
cv2.moveWindow(resultname, displaycoloffset,000)
"""
while True :
    frame = camera.capture()
    #cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    cv2.imshow(resultname, frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
         break
"""
#from scipy.optimize import curve_fit

#fig, ax = plt.plot()
"""
# For Zoom
zoom_level = 1.0

while True:
    frame = camera.capture()  # Capture a frame from the camera
    #Gaussian Fit
     # Convert the frame to grayscale
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Apply thresholding to create a binary image
    _, binary_frame = cv2.threshold(blurred_frame, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours
    for contour in contours:
        # Fit an ellipse to the contour
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            
            # Draw the ellipse on the frame
            frame_with_fit = cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
            
            # Extract ellipse parameters
            center, axes, angle = ellipse
            xo, yo = center
            sigma_x = axes[0] / 2
            sigma_y = axes[1] / 2
            
            # Plot the Gaussian fit on the axes
            gaussian_values = gaussian(x, y, xo, yo, sigma_x, sigma_y)
            ax.imshow(frame_with_fit)
            ax.contour(x, y, gaussian_values, cmap='Reds', levels=5)
            plt.draw()
            plt.pause((0.001), int(yo), (int(sigma_x), int(sigma_y)), 0, 0, 360, (0, 0, 255), 2)
                                        
    # Calculate the zoomed region of interest
    frame_height, frame_width = frame.shape[:2]
    zoomed_width = int(frame_width / zoom_level)
    zoomed_height = int(frame_height / zoom_level)
    x_offset = (frame_width - zoomed_width) // 2
    y_offset = (frame_height - zoomed_height) // 2
    zoomed_frame = frame[y_offset:y_offset+zoomed_height, x_offset:x_offset+zoomed_width]
    # Image Centroid

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        
            # Draw a circle at the centroid point
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Display the zoomed frame
    cv2.imshow("Video", zoomed_frame)

    key = cv2.waitKey(1)
    if key == ord('+'):  # Zoom in
        zoom_level *= 1.1
    elif key == ord('-'):  # Zoom out
        zoom_level /= 1.1
    elif key == ord('q'):  # Quit the program
        break
    elif key == ord('s'):  # Save the current frame
        filename = input("Enter a filename to save the FITS image: ")

        # Save the frame as a FITS file
        hdu = fits.PrimaryHDU(data=np.array(frame, dtype=np.uint16))
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        print("FITS image saved as", filename)

camera.close_camera()
cv2.destroyAllWindows()



'''
#This code didn't work during initial test, but we probably want it eventually, for faster frame capture
print('Enabling video mode')
camera.start_video_capture()

# Set the timeout, units are ms
timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
camera.default_timeout = timeout


print('Capturing a single 8-bit mono frame')
filename = 'image_video_mono.jpg'
camera.set_image_type(asi.ASI_IMG_RAW8)
camera.capture_video_frame(filename=filename)

print('Saved to %s' % filename)
'''
"""
while cap.isOpened():

# Capture frame-by-frame
    ret, frame = cap.read()
    if ret is None:
        break
    
    try:
        frame = imutils.resize(frame, width=750)
    except:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)


if first_frame is None:
    first_frame = gray
next_frame = gray

delay_counter += 1

if delay_counter > 5:
    delay_counter = 0
    first_frame = next_frame

frame_delta = cv2.absdiff(first_frame, next_frame)
thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.dilate(thresh, None, iterations=2)
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)