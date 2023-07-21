import zwoasi as asi
import os
import cv2
import numpy as np
import time
from astropy.io import fits


#Grab camera frames from ZWO cameras
#PMH 6/29/23
#Following example from https://rk.edu.pl/en/scripting-machine-vision-and-astronomical-cameras-python/
#More code also at: https://github.com/python-zwoasi/python-zwoasi/blob/master/zwoasi/examples/zwoasi_demo.py

#Below didn't seem to work.  Eventaully, this is the write way to setup installation of the .dylib file
#env_filename = os.getenv('ZWO_ASI_LIB')
#print (env_filename)

path = '/Users/piramonkumnurdmanee/Documents/lamat-python/venv/lib/python3.10/site-packages/libASICamera2.dylib'		#hardcoded path. Needs to be adjusted on each computer 

asi.init(path)   #The first time this is run you will need to go to your System Preferences under "Security and Privacy" and click "Open Anyway"

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
camera.set_control_value(asi.ASI_EXPOSURE, 3000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0)


#Documentation says the ASI-290mm has 12 bit depth.  But the SDK has no ASI_IMG_RAW12.  Maybe 16 bit is how you get it?
print('Capturing a single 16-bit mono image')

camera.set_image_type(asi.ASI_IMG_RAW16)
filename = 'image_mono16.tiff'
camera.capture(filename=filename)			#save in format for easy display

frame = camera.capture()			#save in format for analysis
# Create a new HDU (Header/Data Unit) object
hdu = fits.PrimaryHDU(data=frame)

# Create a new HDU list and add the HDU to it
hdulist = fits.HDUList([hdu])

# Write the HDU list to a FITS file
hdulist.writeto('testfile.fits',overwrite=True)



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


# For Zoom
# Get the camera's sensor dimensions
sensor_width, sensor_height = camera.get_roi()

# Set the initial ROI (Region of Interest) dimensions to the middle of the frame
roi_width = int(sensor_width / 2)
roi_height = int(sensor_height / 2)

# Calculate the starting X and Y coordinates for the ROI to center it
roi_x = int((sensor_width - roi_width) / 2)
roi_y = int((sensor_height - roi_height) / 2)

# Capture and display the video with the initial ROI
while True:
    # Set the ROI (Region of Interest) for the current frame
    camera.set_roi(roi_x, roi_y, roi_width, roi_height)

    # Capture a frame from the camera
    frame = camera.capture()

    # Display the frame with the current ROI
    cv2.imshow("Zoomed In", frame)

    # Check for user input to zoom in or zoom out
    key = cv2.waitKey(1)
    if key == ord('+'):  # Zoom in
        roi_width = int(roi_width * 0.8)
        roi_height = int(roi_height * 0.8)
        roi_x = int(roi_x + (roi_width * 0.1))
        roi_y = int(roi_y + (roi_height * 0.1))
    elif key == ord('-'):  # Zoom out
        roi_width = int(roi_width * 1.2)
        roi_height = int(roi_height * 1.2)
        roi_x = int(roi_x - (roi_width * 0.1))
        roi_y = int(roi_y - (roi_height * 0.1))
    elif key == ord('q'):  # Quit the program
        break