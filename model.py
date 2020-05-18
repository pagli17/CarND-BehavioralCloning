import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                # flip images
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*(-1))
                
                # multiple cameras with flipping
                correction = 0.2
                left_name = './data/IMG/' + batch_sample[1].split('/')[-1]
                right_name = './data/IMG/' + batch_sample[2].split('/')[-1]
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                images.append(left_image)
                images.append(right_image)
                left_angle = center_angle + 1 * correction
                # The car tends to steer more to the right.
                right_angle = center_angle - 1* correction
                angles.append(left_angle)
                angles.append(right_angle)
                
                #images.append(cv2.flip(left_image,1))
                #angles.append(left_angle*-(1))
                #images.append(cv2.flip(right_image,1))
                #angles.append(right_angle*-(1))
                

                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Model design taken by NVIDIA paper shown as example in the video lessons
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))
print(model.summary())

plt.hist(
#model.compile(loss='mse', optimizer='adam')
#history_object = model.fit_generator(train_generator, samples_per_epoch =
#    len(train_samples), validation_data = 
#    validation_generator,
#    nb_val_samples = len(validation_samples), 
#    nb_epoch=8, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())
#model.save('model_g.h5')

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
#plt.savefig('MSEL_g8.png')
