
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import json
import tensorflow as tf
import random
from pathlib import PurePosixPath
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


cameras = ['left', 'center', 'right']
camera_center = ['center']
steering_offset = {'left': 0.25, 'center': 0., 'right': -.25}


# In[ ]:


#NVIDIA MODEL Uses YUV, Lets try YUV
def load_image(path, filename):
    filename = filename.strip()
    if filename.startswith('IMG'):
        filename = path+'/'+filename
    else:
        filename = path+'/IMG/'+PurePosixPath(filename).name
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def randomise_image_brightness(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
    return yuv

def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)
    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]

def jitter_image_rotation(image, steering):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering

def jitter_camera_image(row, log_path, cameras):
    steering = getattr(row, 'steering')

    # use one of the cameras randomily
    camera = cameras[random.randint(0, len(cameras)-1)]
    steering += steering_offset[camera]

    image = load_image(log_path, getattr(row, camera))
    image, steering = jitter_image_rotation(image, steering)
    image = randomise_image_brightness(image)

    return image, steering

def gen_train_data(path='./data', log_file='driving_log.csv', skiprows=1,
                   cameras=cameras, batch_size=128):

    # load the csv log file
    print("Cameras: ", cameras)
    print("Log path: ", path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right',
                    'steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(path+'/'+log_file,
                          names=column_names, skiprows=skiprows)
    data_count = len(data_df)

    print("Data in Log, %d rows." % (len(data_df)))

    while True:
        features = []
        labels = []
        #Choosing random samples  
        
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]

            image, steering = jitter_camera_image(row, path, cameras)

            if random.random() >= .5 and abs(steering) > 0.1:
                image = cv2.flip(image, 1)
                steering = -steering

            image = crop_camera(image)
            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))


# In[ ]:



def build_nvidia_model(img_height=66, img_width=200, img_channels=3,
                       dropout=.5):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model


# In[ ]:



def main(_):
    epochs = 10
    batch_size = 128
    dropout = 0.55
    data_dir = "./longdata"
    validation_dir = "./udacity_data"
    train_log = pd.read_csv(data_dir+"/driving_log.csv")
    train_size = train_log.shape[0]
    valid_log = pd.read_csv(data_dir+"/driving_log.csv")
    valid_size = valid_log.shape[0]
    # build model and display layers
    model = build_nvidia_model(dropout=dropout)
    for l in model.layers:
        print(l.name, l.input_shape, l.output_shape,
        l.activation if hasattr(l, 'activation') else 'none')
    
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)
    #load previous trained model
    model.load_weights("model.h5")
    model.fit_generator(
        gen_train_data(path=data_dir,
                       cameras=cameras,
                       batch_size=batch_size
                       ),
        samples_per_epoch=train_size,
        nb_epoch=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                  patience=1, verbose=1, mode='auto')],
        validation_data=gen_train_data(path=validation_dir,
                                     batch_size=batch_size),
        nb_val_samples=valid_size)

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

