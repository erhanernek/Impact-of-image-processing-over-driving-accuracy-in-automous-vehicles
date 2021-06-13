import sys
import random
import numpy as np
import logging
import glob
import os
from datetime import datetime
from IPython.display import display, clear_output
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
client.reload_world()
#for mapName in client.get_available_maps():
#    print(mapName)


car = None
camera = None
camera2 = None
camera3 = None
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

root_dir = "/TESTDATA_3camera/" + today.strftime('%Y%m%d_')+ "_"+ h + "_"+ m + "_npy"
#root_dir = "/TESTDATA/" + today.strftime('%Y%m%d_')+ h + m + "_npy"
HEIGHT = 88
WIDTH = 200
print(root_dir)

try:
    os.makedirs(root_dir)
except:
    print(" ")
try:
    inputs_file = open(root_dir + "/inputs_rgb.npy","ba+") 
    inputs_file2 = open(root_dir + "/inputs_depth.npy","ba+") 
    inputs_file3 = open(root_dir + "/inputs_ss.npy","ba+")
    outputs_file = open(root_dir + "/outputs.npy","ba+")     
except:
    print("Error encountered on file opening")
    
walker_blueprint = world.get_blueprint_library().filter("walker.pedestrian.*")
controller_blueprint = world.get_blueprint_library().find('controller.ai.walker')

pedestrians = []  

for i in range(150): 

    pedestrian_transform = carla.Transform()
    pedestrian_transform.location = world.get_random_location_from_navigation()
    pedestrian_transform.location.z += 1

    walker = random.choice(walker_blueprint)
    pedestrian = world.spawn_actor(walker, pedestrian_transform)
    world.wait_for_tick()

    pedestrian_controller = world.spawn_actor(controller_blueprint, carla.Transform(), pedestrian)
    world.wait_for_tick()

    pedestrian_controller.start()
    pedestrian_controller.go_to_location(world.get_random_location_from_navigation())

    pedestrians.append(pedestrian)
    pedestrians.append(pedestrian_controller)
    
    
time.sleep(0.1)

car_bluep = world.get_blueprint_library().find('vehicle.audi.tt')
car_bluep.set_attribute('role_name','ego')
color = random.choice(car_bluep.get_attribute('color').recommended_values)
car_bluep.set_attribute('color',color)

car_location = world.get_map().get_spawn_points()
no_of_car_location = len(car_location)

transform = car_location[6]
car = world.spawn_actor(car_bluep,transform)
print('\nCar is in simulation.')

'''
if 0 < no_of_car_location:
    random.shuffle(car_location)
    transform = car_location[0]
    car = world.spawn_actor(car_bluep,transform)
    print('\nCar is in simulation.')
else: 
    logging.warning('No spawn locations')
    '''  

camera_bluep = None
camera_bluep = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bluep.set_attribute("image_size_y",str(HEIGHT))
camera_bluep.set_attribute("image_size_x",str(WIDTH))
camera_bluep.set_attribute("fov",str(105))
cam_location = carla.Location(2,0,1)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location, cam_rotation)
camera = world.spawn_actor(camera_bluep,cam_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid)

camera_bluep2 = None
camera_bluep2 = world.get_blueprint_library().find('sensor.camera.depth')
camera_bluep2.set_attribute("image_size_y",str(HEIGHT))
camera_bluep2.set_attribute("image_size_x",str(WIDTH))
camera_bluep2.set_attribute("fov",str(105))
cam_location2 = carla.Location(2,0,1)
cam_rotation2 = carla.Rotation(0,0,0)
cam_transform2 = carla.Transform(cam_location2,cam_rotation2)
camera2 = world.spawn_actor(camera_bluep2,cam_transform2,attach_to=car, attachment_type=carla.AttachmentType.Rigid)    

camera_bluep3 = None
camera_bluep3 = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
camera_bluep3.set_attribute("image_size_y",str(HEIGHT))
camera_bluep3.set_attribute("image_size_x",str(WIDTH))
camera_bluep3.set_attribute("fov",str(105))
cam_location3 = carla.Location(2,0,1)
cam_rotation3 = carla.Rotation(0,0,0)
cam_transform3 = carla.Transform(cam_location3,cam_rotation2)
camera3 = world.spawn_actor(camera_bluep3,cam_transform3,attach_to=car, attachment_type=carla.AttachmentType.Rigid)   


#Function to convert image to a numpy array
def image_convert(image):
    frame = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))   
    frame = np.reshape(frame, (image.height, image.width, 4))  
    handled_image = frame[:, :, :3]/255
    return handled_image

#Save required data
def saving(carla_image):
    image = image_convert(carla_image)
    control = car.get_control()
    controls = [control.steer, control.throttle, control.brake]
    np.save(outputs_file, controls)
    np.save(inputs_file, image)
    

def saving2(carla_image):
    image = image_convert(carla_image)
    np.save(inputs_file2, image)


def saving3(carla_image):
    image = image_convert(carla_image)
    np.save(inputs_file3, image)

car.set_autopilot(True)

camera.listen(saving)
camera2.listen(saving2)
camera3.listen(saving3)


try:
    i = 0
    while i < 5000:
        screenshot = world.wait_for_tick()
        clear_output(wait=True)
        display(f"{str(i+1)} frames saved")
        i += 1
except:
    print('\nSimulation error.')
       
if car is not None:
    if camera is not None:
        camera.stop()
        camera.destroy()
    car.destroy()
if pedestrians is not None:
    client.apply_batch([carla.command.DestroyActor(x) for x in pedestrians])

inputs_file.close()
inputs_file2.close()
inputs_file3.close()
outputs_file.close()
print("Done...")
print(root_dir)