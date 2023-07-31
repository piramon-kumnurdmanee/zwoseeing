import zwoasi as asi
import os
import cv2
import numpy as np
import time
import imutils
from astropy.io import fits
import matplotlib.pyplot as plt

#Grab camera frames from ZWO cameras
#PMH 6/29/23
#Following example from https://rk.edu.pl/en/scripting-machine-vision-and-astronomical-cameras-python/
#More code also at: https://github.com/python-zwoasi/python-zwoasi/blob/master/zwoasi/examples/zwoasi_demo.py

#Below didn't seem to work.  Eventaully, this is the write way to setup installation of the .dylib file
#env_filename = os.getenv('ZWO_ASI_LIB')
#print (env_filename)

path = '/Users/piramonkumnurdmanee/Documents/zwoseeing/venv/lib/python3.10/site-packages/libASICamera2.dylib'		#hardcoded path. Needs to be adjusted on each computer 

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
camera.set_control_value(asi.ASI_EXPOSURE, 130000) # microseconds   This saturated the images
camera.set_control_value(asi.ASI_EXPOSURE, 30000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 500)
camera.set_control_value(asi.ASI_FLIP, 0)


#Documentation says the ASI-290mm has 12 bit depth.  But the SDK has no ASI_IMG_RAW12.  Maybe 16 bit is how you get it?




#Create an endless loop to show images from camera
resultname = 'Images'
#setup display of camera results
displaycoloffset = 0000
cv2.namedWindow(resultname)        # Create a window to show captured images
cv2.moveWindow(resultname, displaycoloffset,000)
# Find th star region
# Find the brightest point
# Find Centroid
def find_image_centroid(image):
    start = time.time()
    height, width = image.shape
    sum_x, sum_y, total = 0, 0, 0
    # Find a new way to write new. For loop usually take long time 
    # Centroid Variation, polar
    # get a weight x,y
    x_w, y_w = np.arange(width), np.arange(height)
    # span into 2D
    x_w2d, y_w2d = np.meshgrid(x_w, y_w)
    # multiply and sum
    x_sum, y_sum = np.multiply(image, x_w2d).sum(), np.multiply(image, y_w2d).sum()
    # divide
    x_centroid, y_centroid = x_sum / image.sum(), y_sum / image.sum()

    #print("Time: {}s".format(time.time() - start))

    return x_centroid, y_centroid


prev_centroid_x = None
prev_centroid_y = None
start_time = time.time()
x_centroid = []
y_centroid = []
centroid_distances = []  # List to store centroid distances
timestamps = []



# subimage = image[20:24,31:35], subimage*X
""" 1. Find star routine
    2. Create sub image
    3. Improve center of mass efficiency
"""
# Take it frame by frame faster
"""For short exposure if we take many frames, S.D. tell seeing
    Does it make any different if we take less frame and do it longer. See if it get the same result."""

camera.start_video_capture()
while True:
    frame = camera.capture_video_frame()  # Capture a frame from the camera

    x_centroid, y_centroid = find_image_centroid(frame)

    # Calculate the distance between the current and previous centroid positions using Pythagorean theorem
    if prev_centroid_x is not None and prev_centroid_y is not None:
        diff_x = x_centroid - prev_centroid_x
        diff_y = y_centroid - prev_centroid_y
        distance = np.sqrt(diff_x**2 + diff_y**2)
        centroid_distances.append(distance)

        # Calculate the time elapsed since the starting point
        current_time = time.time()
        time_elapsed = current_time - start_time
        timestamps.append(time_elapsed)

    # Print the centroid coordinates, time elapsed, and centroid distance
    print("Centroid: ({}, {})".format(x_centroid, y_centroid))
    if prev_centroid_x is not None and prev_centroid_y is not None:
        print("Time Elapsed:", time_elapsed, "s")
        print("Centroid Distance:", distance)

    # Update the previous centroid positions for the next iteration
    prev_centroid_x = x_centroid
    prev_centroid_y = y_centroid

    

    #Print the centroid coordinates
    #print("Centroid X:", centroid_x)
    #print("Centroid Y:", centroid_y)



    # Display the zoomed frame
    cv2.imshow("Video", frame)

    # Update the plot with the centroid distances over time
    plt.scatter(timestamps, centroid_distances)
    plt.ylim(-0.5,0.5)
    plt.xlabel('Time')
    plt.ylabel('Centroid Distance')
    plt.title('Centroid Distance between Consecutive Frames')
    plt.pause(0.001)
    plt.clf()

    key = cv2.waitKey(1)
   
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('s'):  # Save the current frame
        filename = input("Enter a filename to save the fits image: ")

        # Save the frame as a FITS file
        hdu = fits.PrimaryHDU(data=np.array(frame, dtype=np.uint16))
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        print("fits image saved as", filename)
   
   
    """    
    time = np.arange(len(centroid_distances))
    plt.scatter(time, centroid_distances)
    plt.xlabel('Time')
    plt.ylabel('Centroid Position')
    plt.title('Centroid Position over Time')
    plt.legend()
    plt.draw()
    plt.pause(0.001)
    plt.clf()
    """
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