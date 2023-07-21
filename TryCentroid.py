import zwoasi as asi
import os
import cv2
import numpy as np
import time
import imutils
from astropy.io import fits
import matplotlib as plt

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
camera.set_control_value(asi.ASI_EXPOSURE, 8000) # microseconds		Nominal value
camera.set_control_value(asi.ASI_WB_B, 99)
camera.set_control_value(asi.ASI_WB_R, 75)
camera.set_control_value(asi.ASI_GAMMA, 50)
camera.set_control_value(asi.ASI_BRIGHTNESS, 50)
camera.set_control_value(asi.ASI_FLIP, 0)
#Create an endless loop to show images from camera
resultname = 'Images'
#setup display of camera results
displaycoloffset = 0000
cv2.namedWindow(resultname)        # Create a window to show captured images
cv2.moveWindow(resultname, displaycoloffset,000)

while True:
    frame = camera.capture()  # Capture a frame from the camera
    cv2.imshow("Video", frame)
# Save the frame as a FITS file
""" key = cv2.waitKey(1)
    if key == ord('s'):  # Save the current frame
        filename = input("Enter a filename to save the FITS image: ")
    hdu = fits.PrimaryHDU(data=np.array(frame, dtype=np.uint16))
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)
    print("FITS image saved as", filename)"""


camera.close_camera()
cv2.destroyAllWindows()