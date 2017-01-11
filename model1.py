import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2, activity_l2
import h5py
import numpy as np

img_width, img_height = 128,128

train_data_dir = '../train'
validation_data_dir = '../validation'

nb_train_samples = 202249
nb_validation_samples = 2688
nb_epoch = 5

model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=(3, img_width,img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.8))
model.add(Dense(512, W_regularizer = l2(0.001), activity_regularizer = activity_l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.8))

#model.add(Dense(1024, W_regularizer = l2(0.001), activity_regularizer = activity_l2(0.001)))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))


sgd  = keras.optimizers.SGD(lr=0.0625,decay = 1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer= 'adadelta', metrics=['accuracy'])
model.load_weights('/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/weights3.02-0.63.hdf5')


############

train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.3,height_shift_range=0.3,shear_range=0.3,channel_shift_range=0.2,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
         train_data_dir,
         target_size=(img_width, img_height),
         batch_size=128,shuffle=True,classes=["ns","ew","flat","other"])

train_datagen.fit(train_generator)

validation_generator = test_datagen.flow_from_directory(
         validation_data_dir,
         target_size=(img_width, img_height),
         batch_size=128,shuffle=True,classes=["ns","ew","flat","other"])

filepath = '/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/weights_lastday.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

model.fit_generator(
       train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, verbose = 1, callbacks = [checkpoint])

model.save_weights('/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/weights_lastday.h5')


