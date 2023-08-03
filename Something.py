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
from astropy.visualization import SqrtStretch, ImageNormalize

# Grab camera frames from ZWO cameras
path = '/Users/piramonkumnurdmanee/Documents/zwoseeing/venv/lib/python3.10/site-packages/libASICamera2.dylib'		#hardcoded path. Needs to be adjusted on each computer 
asi.init(path)

num_cameras = asi.get_num_cameras()
if num_cameras == 0:
    raise ValueError('No cameras found')

camera_id = 0  # Use the first camera from the list
camera = asi.Camera(camera_id)
camera_info = camera.get_camera_property()
print(camera_info)

# Get all of the camera controls and print them
controls = camera.get_controls()
for cn in sorted(controls.keys()):
    print('    %s:' % cn)
    for k in sorted(controls[cn].keys()):
        print('        %s: %s' % (k, repr(controls[cn][k])))

# Use minimum USB bandwidth permitted
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue'])

# Set some sensible defaults for camera settings
camera.disable_dark_subtract()

camera.set_control_value(asi.ASI_GAIN, 100)
camera.set_control_value(asi.ASI_EXPOSURE, 30000)  # Set the exposure time in microseconds (adjust as needed)
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0)

# Start video capture
camera.start_video_capture()

def find_centroid(image):
    height, width = image.shape
    x_w, y_w = np.arange(width), np.arange(height)
    x_w2d, y_w2d = np.meshgrid(x_w, y_w)
    x_sum, y_sum = np.multiply(image, x_w2d).sum(), np.multiply(image, y_w2d).sum()
    x_centroid, y_centroid = x_sum / image.sum(), y_sum / image.sum()
    return x_centroid, y_centroid


start_time = time.time()
x_centroid = []
y_centroid = []
x_centroid_sub = []
y_centroid_sub = []
prev_centroid_x = []
prev_centroid_y = []
timestamps = []
centroid_distances = []
diff_x = []
diff_y = []
# Find the brightest star before entering the loop
brightest_source_x, brightest_source_y = None, None
star_radius = 20  # Radius of the circle around the brightest star

# Capture an initial frame for finding the brightest star
frame = camera.capture_video_frame()
mean, median, std = sigma_clipped_stats(frame, sigma=3.0)
threshold = 5.0 * std
tbl = DAOStarFinder(fwhm=3.0, threshold=threshold)(frame - median)
# tbl what is it?

if tbl is not None and len(tbl) > 0:
    brightest_source_idx = np.argmax(tbl['peak'])
    brightest_source_x = tbl['xcentroid'][brightest_source_idx]
    brightest_source_y = tbl['ycentroid'][brightest_source_idx]

camera.start_video_capture()
while True:
    frame = camera.capture_video_frame()  # Capture a frame from the camera

    if brightest_source_x is not None and brightest_source_y is not None:
        # Extract a subimage centered on the brightest star
        subimage_radius = 50  # Adjust this value to change the size of the subimage
        subimage_x = int(brightest_source_x)
        subimage_y = int(brightest_source_y)
        subimage = frame[subimage_y - subimage_radius:subimage_y + subimage_radius,
                         subimage_x - subimage_radius:subimage_x + subimage_radius]

        # Calculate the centroid of the subimage
        x_centroid_sub, y_centroid_sub = find_centroid(subimage)

        # Display the zoomed subimage
        norm = ImageNormalize(stretch=SqrtStretch()) 
        plt.imshow(subimage, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
        plt.title("Zoomed Subimage (Centroid)")
        plt.pause(0.001)

    # Update the centroid positions for the next iteration
    if np.isfinite(x_centroid_sub) and np.isfinite(y_centroid_sub):
        x_centroid.append(x_centroid_sub)
        y_centroid.append(y_centroid_sub)

    # Calculate the distance between the current and previous centroid positions using Pythagorean theorem
    if prev_centroid_x is not None and prev_centroid_y is not None:
        diff_x = x_centroid_sub - prev_centroid_x
        diff_y = y_centroid_sub - prev_centroid_y
        distance = np.sqrt(diff_x**2 + diff_y**2)
        centroid_distances.append(distance)
        # Calculate the time elapsed since the starting point
        current_time = time.time()
        time_elapsed = current_time - start_time
        timestamps.append(time_elapsed)

    # Print the centroid coordinates, time elapsed, and centroid distance
    print("Centroid: ({}, {})".format(x_centroid_sub, y_centroid_sub))
    if prev_centroid_x is not None and prev_centroid_y is not None:
        print("Time Elapsed:", time_elapsed, "s")
        print("Centroid Distance:", distance)

    # Update the previous centroid positions for the next iteration
    prev_centroid_x = x_centroid_sub
    prev_centroid_y = y_centroid_sub

    # Display the zoomed frame with apertures
    positions = [(brightest_source_x, brightest_source_y)]
    apertures = CircularAperture(positions, r=star_radius)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.imshow(frame, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)

    # Draw a circle around the brightest star
    circle = plt.Circle((brightest_source_x, brightest_source_y), star_radius, color='red', fill=False, lw=2)
    plt.gca().add_patch(circle)

    plt.title("Video Tracking - Press 'q' to stop")
    plt.pause(0.001)

    cv2.imshow("Video", frame)

    # Check if the user pressed 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Stop video capture and close the camera
camera.stop_video_capture()
camera.close_camera()

# Plot the centroid distances over time
plt.scatter(timestamps, centroid_distances)
plt.ylim(-0.5, 0.5)
plt.xlabel('Time')
plt.ylabel('Centroid Distance')
plt.title('Centroid Distance between Consecutive Frames')
plt.show()