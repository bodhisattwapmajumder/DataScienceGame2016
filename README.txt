
************************ Data Preparation   *********************************

Train data consists of 8,000 images, test data consists of 13,999 test images given in the problem. 
All images have been resized to 128 x 128. (see resized_images.py)

Unannotated data have been predicted by intermediate model and kept in the proper directory (class). [Those examples for which the model had more than 80% confident]

All north-south images has been rotated 90 degree to create augmented east-west images and vice versa.
After having all the images of these four classes we have applied several transformations to generate more number of training images- 
1. 3 random crops to all NS and EW images
2. 2 random crops and +30 degree and -30 degree rotation
3. Blurred, tilted (very small degree of rotation) and flipped (horizontally and vertically) images have been used for different phase of model building(See image_preprocessing.py)

After the augmentation, these 2,02,249 images have been fed into two models.
EW - 57,074
NS - 57,841
Flat - 46,285
Other - 41,049


************************   Model 1 architecture *********************************
We used keras deep learning package to train our convolution neural networks. The architecture of the model 1 was -


conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d
=> flatten => dense(512), dropout(0.5) => dense(4), softmax

(see model1.py)

All convolution layers have 64 filters and l2 regularization has been used.

************************   Model 2 architecture *********************************

conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d => conv2d, relu, maxpooling2d
=> flatten => dense(1024), dropout(0.5) => dense(1024), dropout(0.5) => dense(4), softmax

(see model2.py)

All convolution layers have 64 filters and l2 regularization has been used.

*************************   Testing ************************************

For testing see model1_test.py and model2_test.py


************************* Final result **********************************

We have taken out 30 models based on low validation loss and make an ensemble of equal weights and achieved test accuracy of 82.854%. 

All the model outputs include probabilities of an image belonging to each classes. All probabilities (30 models) have been added (in excel) and based on the maximum probability the final class has been predicted (in excel). For example the final calculation of the ensembles 
see - best_solution_ensemble.csv 

