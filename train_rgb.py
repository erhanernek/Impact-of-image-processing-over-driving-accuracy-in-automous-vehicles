import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.optimizers import SGD
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.compat.v1.keras.backend import set_session


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
  try:
    for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


#Previous model to continue training
#load_previous = None
load_previous = f"/MODELS_rgb/First_model.13.testing.h5"
model_adi = "14"
#Variables
EPOCHS=10
MEMORY_FRACTION = 0.6
TRAINING_BATCH_SIZE = 16
DATASET = "20210420__22_06_npy"
DIRECTORY = f"/TESTDATA_rgb/{DATASET}"
HEIGHT = 88
WIDTH = 200
MODEL_NAME = "Xception"

#Optimizers
OPT = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#OPT = keras.optimizers.Adam(learning_rate=0.001)/7

#We open the training data
INPUTS_FILE_rgb = open(DIRECTORY + "/inputs_rgb.npy","br") 
OUTPUTS_FILE = open(DIRECTORY + "/outputs.npy","br")  

#We get the data in
inputs_rgb = []
outputs = []


while True:   
    try:
        input_rgb = np.load(INPUTS_FILE_rgb)
        inputs_rgb.append(input_rgb) 
    except:
        break

while True:
    try:
        output = np.load(OUTPUTS_FILE)
        outputs.append(output)
    except: 
        break


#input_np = np.array(inputs)
output_np = np.array(outputs)

input_rgb_np = np.array(inputs_rgb)


#we close everything
inputs = None
outputs = None

INPUTS_FILE_rgb.close()

OUTPUTS_FILE.close()


input_rgb_np = input_rgb_np[300:,:,:]
output_np = output_np[300:,:]
'''
#Let's print some metrics
print("------------------------------------------------------------------------")

print('final_input_np shape:')
print(final_input_np.shape) 
print("------------------------------------------------------------------------")
    
print("Output Shape")
print(output_np.shape)
print("-------------------------------------------------------------------------------------")
'''

with tf.device('/gpu:0'):
    trainx, testx, trainy, testy = train_test_split(input_rgb_np, output_np)
    
    
with tf.device('/gpu:0'):
    if load_previous is not None:
        model = models.load_model(load_previous)
        print(f"loaded model:{load_previous}")
    else:
        '''
        #Spintronics model
        model = models.Sequential()
        model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, 9)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='sigmoid'))
        model.add(layers.Dense(3))

        '''
        #JDO Model
        base_model= Xception(weights=None, include_top=False, input_shape=(HEIGHT, WIDTH,3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #3 actions, 3 predictions left, right, straight
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs = base_model.input, outputs = predictions)      
        

with tf.device('/gpu:0'):
    if load_previous is None:
        #model.compile(optimizer=OPT, loss="categorical_crossentropy", metrics=['accuracy'])
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    my_callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        #tf.keras.callbacks.ModelCheckpoint(filepath='models/model.' + DATASET + '.{epoch:02d}.{val_accuracy:.2f}-{val_loss:.2f}.h5', save_weights_only=False),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
        
#my_callbacks = []

with tf.device('/gpu:0'):
    history = model.fit(trainx, trainy, epochs=EPOCHS, validation_data=(testx, testy), callbacks=my_callbacks)
    #history = model.fit(trainx, trainy, epochs=EPOCHS, batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=my_callbacks)

test_loss, test_acc = model.evaluate(testx, testy, verbose=2)
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print(test_loss, test_acc)
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
print("--------------------------------------------------------------")

#model_file = f"MODELS1/workshop_model.{DATASET}.{test_acc:.2f}-{test_loss:.2f}.h5"

model_file = f"/MODELS_rgb/rgb_camera_model.{model_adi}.testing.h5"
model.save(model_file)
print(model_file)
