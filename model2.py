import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2, activity_l2
import h5py
import numpy as np

img_width, img_height = 128,128

train_data_dir = '/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/AllDataAugment'
validation_data_dir = '/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/cropNblur/validation'
# # test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test'
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

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dropout(0.5))
model.add(Dense(1024, W_regularizer = l2(0.001), activity_regularizer = activity_l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024, W_regularizer = l2(0.001), activity_regularizer = activity_l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Dense(1024, W_regularizer = l2(0.001), activity_regularizer = activity_l2(0.001)))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))


sgd  = keras.optimizers.SGD(lr=0.0625,decay = 1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer= 'adadelta', metrics=['accuracy'])
model.load_weights('/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/large/weights_hori_shift.04-0.55.hdf5')


# ##########

train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=0.3,height_shift_range=0.3,shear_range=0.2,zoom_range=0.4,vertical_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen2 = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
         train_data_dir,
         target_size=(img_width, img_height),
         batch_size=32,shuffle=True,classes=["ns","ew","flat","other"])

validation_generator = test_datagen.flow_from_directory(
         validation_data_dir,
         target_size=(img_width, img_height),
         batch_size=32,shuffle=True,classes=["ns","ew","flat","other"])

# # test_generator = test_datagen2.flow_from_directory(
# #     test_data_dir,
# #     target_size =(img_width, img_height),
# #     batch_size = 64,class_mode = None,shuffle=False 
# #     )


filepath = '/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/large/weights_lastday.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

model.fit_generator(
       train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, verbose = 1, callbacks = [checkpoint])


# a = model.evaluate_generator(test_generator,896,1)
# print a

#np.savetxt("PredProb_validation_set_as_test.csv",a,delimiter=",")

model.save_weights('/Users/sourav/Documents/dsg_challenge-BodhiRobinJayantaAyan/all_augment_models/large/weights_lastday.hdf5')

# # json_string = model.to_json()

#model.load_weights('weights_128_image_size.h5')

########
"""predict on the validation set"""
# test_data_dir = '/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test/val_test'
# output_file = open("/Users/ROBIN/Desktop/dsg/first_resize_train_test_data/val_test_prediction.txt","w+")
# import os
# from os import listdir
# from os.path import isfile,join
# from PIL import Image
# # import Image
# os.chdir(test_data_dir)
# cwd = os.getcwd()
# files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
# print files_list

# loop_count = 0
# for file in files_list:
#     print file
#     loop_count += 1
#     # if loop_count >10:
#         # break
#     if file[-4:] != ".jpg":
#         continue
#     image_array = np.asarray(Image.open(file))
#     image_array=np.swapaxes(image_array,0,2)
#     image_array=np.swapaxes(image_array,1,2)
#     # print image_array
#     reshaped = image_array[np.newaxis,...]
#     # print reshaped
#     # print image_array.shape,image_array.size
#     # print reshaped.shape
#     a = model.predict(reshaped,batch_size=1)[0]
#     output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
#     output_file.writelines(output_line)
#     print a 

"""predict on the test set"""


#test_data_dir = '/Users/ROBIN/Desktop/dsg/resize_to_128_train_val/test'
#output_file = open("/Users/ROBIN/Desktop/dsg/resize_to_128_train_val/test_prediction.txt","w+")
#import os
#from os import listdir
#from os.path import isfile,join
#from PIL import Image
# import Image
#os.chdir(test_data_dir)
#cwd = os.getcwd()
#files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
#print files_list

#loop_count = 0
#for file in files_list:
#    print file
#    loop_count += 1
#    # if loop_count >10:
#        # break
#    if file[-4:] != ".jpg":
#        continue
#    image_array = np.asarray(Image.open(file))
#    image_array=np.swapaxes(image_array,0,2)
#    image_array=np.swapaxes(image_array,1,2)
#    reshaped = image_array[np.newaxis,...]
#    a = model.predict(reshaped,batch_size=1)[0]
#   output_line = ((str(file[:-4])+","+str(a[0])+","+str(a[1])+","+str(a[2])+","+str(a[3]))+"\n")
#    output_file.writelines(output_line)
#    print a 

