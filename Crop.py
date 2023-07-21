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
camera.set_control_value(asi.ASI_EXPOSURE, 500) # microseconds		Nominal value
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

# Initialize the ZWO ASI290MM camera
camera = ASI_Camera(0)  # Replace 0 with the camera index if multiple cameras are connected
camera.init_camera()

# Get the camera's frame dimensions
frame_width = camera.camera_info['MaxWidth']
frame_height = camera.camera_info['MaxHeight']

# Calculate the starting position for the FOV
start_x = int((frame_width - fov_width) / 2)
start_y = int((frame_height - fov_height) / 2)

# Set the ROI (Region of Interest) for the desired FOV
camera.set_roi(start_x, start_y, fov_width, fov_height)

# Capture and display the video with the limited FOV
while True:
    frame = camera.capture()

    # Display the frame with the limited FOV
    frame.show()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
camera.close_camera()
cv2.destroyAllWindows()