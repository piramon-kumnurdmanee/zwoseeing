import zwoasi as asi
import os
import cv2
import numpy as np
import time
import imutils
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

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
camera.set_control_value(asi.ASI_HIGH_SPEED_MODE, 1) 
camera.set_control_value(asi.ASI_EXPOSURE, 130000) # microseconds   This saturated the images
camera.set_control_value(asi.ASI_EXPOSURE, 8000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0)


#Documentation says the ASI-290mm has 12 bit depth.  But the SDK has no ASI_IMG_RAW12.  Maybe 16 bit is how you get it?

# Create an endless loop to capture video and track the brightest source
resultname = 'Video Tracking'
cv2.namedWindow(resultname)  # Create a window to show captured images

star_tracker = None # At the beginning of the loop, star_tracker is set to None, indicating that no bright source has been detected yet.
#When the DAOStarFinder algorithm successfully detects bright sources in the frame, the code calculates the centroid position of the brightest source and stores it in the brightest_source_x and brightest_source_y variables.
#The star_tracker is then updated with the position of the brightest source. This is achieved by assigning the (brightest_source_x, brightest_source_y) tuple to star_tracker.
star_radius = 20  # Radius of the circle around the brightest source

# Lists to store the centroid positions over time
centroid_x_values = []
centroid_y_values = []

fig, axes = plt.subplots(1, 2, figsize=(10,5))

def plot_centroid(centroid_x_values, centroid_y_values):
    plt.scatter(centroid_x_values, centroid_y_values, marker='o', color='b')
    plt.title('Motion of Centroid Over Time')
    plt.pause(0.001)
    # plt.clf()

camera.start_video_capture()
while True:

    frame = camera.capture_video_frame() # Capture a frame from the camera
    mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
    threshold = 5.0 * std
    tbl = DAOStarFinder(fwhm=3.0, threshold=threshold)(frame - median)
    print(std)

    # Check if sources were found in the frame
    if tbl is not None and len(tbl) > 0:
        positions = np.transpose((tbl['xcentroid'], tbl['ycentroid']))
        apertures = CircularAperture(positions, r=4.0)

        # Find the brightest source
        brightest_source_idx = np.argmax(tbl['peak'])
        brightest_source_x = tbl['xcentroid'][brightest_source_idx]
        brightest_source_y = tbl['ycentroid'][brightest_source_idx]

        # Display the image with apertures
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(frame, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
        apertures.plot(color='blue', lw=1.5, alpha=0.5)

        # Draw a circle around the brightest source
        circle = plt.Circle((brightest_source_x, brightest_source_y), star_radius, color='red', fill=False, lw=2)
        plt.gca().add_patch(circle)

        plt.title("Video Tracking - Press 'q' to stop")
        #plt.pause(0.001)

       # Update star_tracker with the position of the brightest source
        star_tracker = (brightest_source_x, brightest_source_y)

        # Store the centroid positions for plotting
        centroid_x_values.append(brightest_source_x)
        centroid_y_values.append(brightest_source_y)
        plot_centroid(centroid_x_values, centroid_y_values)

        plt.pause(0.001)



    else:
        # If no sources were found, you can display the frame without apertures or circles.
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(frame, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
        plt.title("No Bright Sources Found - Press 'q' to stop")
        plt.pause(0.001)
    
    cv2.imshow(resultname, frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')
    #plot_centroid(centroid_x_values, centroid_y_values)
    # plt.clf()

camera.close_camera()
plt.ioff()
plt.show()  # Show the final centroid plot when the loop is exited
cv2.destroyAllWindows()


# 300 frames then find the Standard Deviation
