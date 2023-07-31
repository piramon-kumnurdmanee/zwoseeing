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
from Camera import *

camera = getCamera()
camera = setCamera(camera, [
    # Use minimum USB bandwidth permitted
    (asi.ASI_BANDWIDTHOVERLOAD, camera.get_controls()['BandWidth']['MinValue']),
    (asi.ASI_GAIN, 100),
    (asi.ASI_HIGH_SPEED_MODE, 1),
    (asi.ASI_EXPOSURE, 130000), # microseconds   This saturated the images
    (asi.ASI_EXPOSURE, 8000), # microseconds		Nominal value
    (asi.ASI_WB_B, 99),
    (asi.ASI_WB_R, 75),
    (asi.ASI_GAMMA, 50),
    (asi.ASI_BRIGHTNESS, 50),
    (asi.ASI_FLIP, 0)
    ])

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

while True:
    frame = camera.capture()  # Capture a frame from the camera
    data = frame  # The frame represents a grayscale image

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = 5.0 * std
    tbl = DAOStarFinder(fwhm=3.0, threshold=threshold)(data - median)
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
        plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
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


    else:
        # If no sources were found, you can display the frame without apertures or circles.
        norm = ImageNormalize(stretch=SqrtStretch())
        plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
        plt.title("No Bright Sources Found - Press 'q' to stop")
        plt.pause(0.001)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Quit the program
        break

    plt.figure()
    plt.plot(centroid_x_values, centroid_y_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')
    plt.title('Motion of Centroid Over Time')
    plt.show()
    plt.pause(0.001)
    plt.clf()

camera.close_camera()
plt.close()
plt.ioff()

# %%
from matplotlib import pyplot as plt
plt.plot([1, 2, 3], [2, 3, 4])
plt.title(r"$\frac{x}{y}$")
# %%

# 300 frames then find the Standard Deviation
