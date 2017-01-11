import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2, activity_l2
import h5py
import numpy as np

img_width, img_height = 128,128

test_data_dir = '../test/'
output_file = open("../final_model","w+")
import os
from os import listdir
from os.path import isfile,join
from PIL import Image
# import Image
os.chdir(test_data_dir)
cwd = os.getcwd()
files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
print files_list

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

model.add(Dense(4))
model.add(Activation('softmax'))


sgd  = keras.optimizers.SGD(lr=0.0625,decay = 1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.load_weights('../weights.00-0.51.hdf5')

loop_count = 0
for file in files_list:
    print loop_count
    loop_count += 1
    if file[-4:] != ".jpg":
        continue
    image_array = np.asarray(Image.open(file))
    image_array=np.swapaxes(image_array,0,2)
    image_array=np.swapaxes(image_array,1,2)
    reshaped = 1.*image_array[np.newaxis,...]/255
    a = model.predict_proba(reshaped,batch_size=1)[0]
    output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
    output_file.writelines(output_line)
