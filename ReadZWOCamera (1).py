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

path = '/Users/piramonkumnurdmanee/Documents/lamat-python/venv/lib/python3.10/site-packages/libASICamera2.dylib'	#hardcoded path. Needs to be adjusted on each computer 

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

camera.set_control_value(asi.ASI_GAIN, 150)
camera.set_control_value(asi.ASI_EXPOSURE, 1300000) # microseconds   This saturated the images
camera.set_control_value(asi.ASI_EXPOSURE, 1300) # microseconds		Nominal value
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


#Create an endless loop to show images from camera
resultname = 'Images'
#setup display of camera results
displaycoloffset = 0000
cv2.namedWindow(resultname)        # Create a window to show captured images
cv2.moveWindow(resultname, displaycoloffset,000)
while True :
    frame = camera.capture()
    cv2.imshow(resultname, frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
         break








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