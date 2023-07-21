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
camera.set_control_value(asi.ASI_EXPOSURE, 15000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
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
    height, width = image.shape
    sum_x, sum_y, total = 0, 0, 0

    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            sum_x += x * pixel
            sum_y += y * pixel
            total += pixel

    centroid_x = sum_x / total
    centroid_y = sum_y / total

    return centroid_x, centroid_y

centroid_x_values = []
centroid_y_values = []

# For Zoom
zoom_level = 1.0

#Take it frame by frame faster
"""For short exposure if we take many frames, S.D. tell seeing
Does it make any different if we take less frame and do it longer. See if it get the same result."""

while True:
    frame = camera.capture()  # Capture a frame from the camera
   
    # Calculate the zoomed region of interest
    frame_height, frame_width = frame.shape[:2]
    zoomed_width = int(frame_width / zoom_level)
    zoomed_height = int(frame_height / zoom_level)
    x_offset = (frame_width - zoomed_width) // 2
    y_offset = (frame_height - zoomed_height) // 2
    zoomed_frame = frame[y_offset:y_offset+zoomed_height, x_offset:x_offset+zoomed_width]
    """
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain brightest areas
    _, thresholded = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of the thresholded areas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if len(contours) > 0:
        brightest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(brightest_contour)

        # Calculate centroid of the brightest area
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']

        # Append the centroid x and centroid y values to the lists
        centroid_x_values.append(centroid_x + x_offset)
        centroid_y_values.append(centroid_y + y_offset)

        # Draw a green circle at the centroid position
        cv2.circle(frame, (int(centroid_x + x_offset), int(centroid_y + y_offset)), 5, (0, 255, 0), -1)
    """
    # Find the centroid of the zoomed frame
    centroid_x, centroid_y = find_image_centroid(zoomed_frame)

    # Draw a green circle at the centroid position
    centroid_x = int(centroid_x) + x_offset
    centroid_y = int(centroid_y) + y_offset
    cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0 , 0), -1)
    
    # Draw a bounding box around the object based on centroid position
    box_size = 50  # Size of the bounding box
    x1 = centroid_x - box_size // 2
    y1 = centroid_y - box_size // 2
    x2 = centroid_x + box_size // 2
    y2 = centroid_y + box_size // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Append the centroid x and centroid y values to the lists
    centroid_x_values.append(centroid_x)
    centroid_y_values.append(centroid_y)

    # Print the centroid coordinates
    print("Centroid X:", centroid_x)
    print("Centroid Y:", centroid_y)



    # Display the zoomed frame
    cv2.imshow("Video", zoomed_frame)
    
    plt.ion()
    """
    plt.scatter(centroid_x_values, centroid_y_values, c='red')
    plt.xlim(450,600)
    plt.ylim(950,1100)
    plt.xlabel('Centroid X')
    plt.ylabel('Centroid Y')
    plt.title('Centroid Position over Time')
    plt.show()
    #plt.pause(0.001)
    #plt.clf()
    """
    time = np.arange(len(centroid_x_values))
    plt.scatter(time, centroid_x_values, label='Centroid X')
    plt.scatter(time, centroid_y_values, label='Centroid Y')
    plt.xlabel('Time')
    plt.ylabel('Centroid Position')
    plt.title('Centroid Position over Time')
    plt.legend()
    plt.draw()
    plt.pause(0.001)
    plt.clf()


    key = cv2.waitKey(1)
    if key == ord('+'):  # Zoom in
        zoom_level *= 1.1
    elif key == ord('-'):  # Zoom out
        zoom_level /= 1.1
    elif key == ord('q'):  # Quit the program
        break
    elif key == ord('s'):  # Save the current frame
        filename = input("Enter a filename to save the fits image: ")

        # Save the frame as a FITS file
        hdu = fits.PrimaryHDU(data=np.array(frame, dtype=np.uint16))
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        print("fits image saved as", filename)
   

# Plot the centroid x and centroid y values over time
#plt.plot(centroid_x_values, centroid_y_values)
#plt.xlabel('Centroid X')
#plt.ylabel('Centroid Y')
#plt.title('Centroid Position over Time')
#plt.show()

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