import zwoasi as asi

def getCamera():
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
    return camera

def setCamera(camera, settings):
    # Set some sensible defaults. They will need adjusting depending upon
    # the sensitivity, lens and lighting conditions used.
    camera.disable_dark_subtract()

    for setting in settings:
        camera.set_control_value(setting[0], setting[1])
    return camera
