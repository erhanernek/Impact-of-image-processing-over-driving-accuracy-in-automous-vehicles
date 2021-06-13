#Dependencies
import glob
import os
import sys
import time
import numpy as np
from IPython.display import display, clear_output
import logging
import random
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Creating a client
client = carla.Client("localhost", 2000)
#client = carla.Client("52.232.96.45", 2000)
client.set_timeout(10.0)
client.reload_world()
for mapName in client.get_available_maps():
    print(mapName)



ego_vehicle = None
ego_cam = None
ego_cam2 = None
ego_cam3 = None
world = client.get_world()
# world.set_weather(carla.WeatherParameters())

today = datetime.now()
if today.hour < 10:
    h = "0"+ str(today.hour)
else:
    h = str(today.hour)
if today.minute < 10:
    m = "0"+str(today.minute)
else:
    m = str(today.minute)
'''
#directory = "data/" + today.strftime('%Y%m%d_')+ h + m + "_npy"

directory = "/TESTDATA/" + today.strftime('%Y%m%d_')+ h + m + "_npy"

print(directory)
'''

WIDTH = 800
HEIGHT = 600
'''
try:
    os.makedirs(directory)
except:
    print("Directory already exists")
try:
    inputs_file = open(directory + "/inputs.npy","ba+") 
    inputs_file2 = open(directory + "/inputs2.npy","ba+") 
    outputs_file = open(directory + "/outputs.npy","ba+")     
except:
    print("Files could not be opened")
 '''   


#Spawn vehicle
ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
ego_bp.set_attribute('role_name','ego')
ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
ego_bp.set_attribute('color',ego_color)

spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    ego_transform = spawn_points[0]
    ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
    print('\nVehicle spawned')
else: 
    logging.warning('Could not found any spawn points')
     
#Adding a RGB camera sensor
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute("image_size_x",str(WIDTH))
cam_bp.set_attribute("image_size_y",str(HEIGHT))
cam_bp.set_attribute("fov",str(105))
cam_location = carla.Location(2,0,1)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location,cam_rotation)
ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
ego_cam.listen(lambda image: image.save_to_disk('output/rgb/%d064.png'%image.frame))

cam_bp2 = None
cam_bp2 = world.get_blueprint_library().find('sensor.camera.depth')
cam_bp2.set_attribute("image_size_x",str(WIDTH))
cam_bp2.set_attribute("image_size_y",str(HEIGHT))
cam_bp2.set_attribute("fov",str(105))
cam_location2 = carla.Location(2,0,1)
cam_rotation2 = carla.Rotation(0,0,0)
cam_transform2 = carla.Transform(cam_location2,cam_rotation2)
ego_cam2 = world.spawn_actor(cam_bp2,cam_transform2,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)       
ego_cam2.listen(lambda image1: image1.save_to_disk('output/depth/%d064.png'%image1.frame,carla.ColorConverter.LogarithmicDepth))

cam_bp3 = None
cam_bp3 = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
cam_bp3.set_attribute("image_size_x",str(WIDTH))
cam_bp3.set_attribute("image_size_y",str(HEIGHT))
cam_bp3.set_attribute("fov",str(105))
cam_location3 = carla.Location(2,0,1)
cam_rotation3 = carla.Rotation(0,0,0)
cam_transform3 = carla.Transform(cam_location3,cam_rotation3)
ego_cam3 = world.spawn_actor(cam_bp3,cam_transform3,attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)       
ego_cam3.listen(lambda image2: image2.save_to_disk('output/ss/%d064.png'%image2.frame,carla.ColorConverter.CityScapesPalette))
#Function to convert image to a numpy array
'''
def process_image(image):
    #Get raw image in 8bit format
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #Reshape image to RGBA
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))
    #Taking only RGB
    processed_image = raw_image[:, :, :3]/255
    return processed_image

#Save required data
def save_image(carla_image):
    image = process_image(carla_image)
    ego_control = ego_vehicle.get_control()
    data = [ego_control.steer, ego_control.throttle, ego_control.brake]
    np.save(inputs_file, image)
    np.save(outputs_file, data)

def save_data(carla_image):
	ego_control = ego_vehicle.get_control()
    data = [ego_control.steer, ego_control.throttle, ego_control.brake]
    np.save(outputs_file, data)
'''
   
#enable auto pilot
ego_vehicle.set_autopilot(True)

#Attach event listeners
#ego_cam.listen(save_data)

try:
    i = 0
    while i < 1200:
        world_snapshot = world.wait_for_tick()
        clear_output(wait=True)
        display(f"{str(i)} frames saved")
        i += 1
except:
    print('\nSimulation error.')
        
if ego_vehicle is not None:
    if ego_cam is not None:
        ego_cam.stop()
        ego_cam.destroy()
    ego_vehicle.destroy()
    '''
inputs_file.close()
outputs_file.close()
print("Data retrieval finished")

print(directory)
'''