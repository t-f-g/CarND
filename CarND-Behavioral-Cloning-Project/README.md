#Project Overview

The requirements of this project were to teach a neural network to drive a virtual track by cloning the behaviour of a human driver.

This project will analyze the output of a driving simulation and predict the appropriate steering angle.  The overall flow will look like the following:

![Overview](project overview.png)

##Approach

The first step before you can create a model architecture is to analyze the actual data and what kind of output you want.  The only output we want is a steering angle, that is clear.  The input data looks complex at first, a large 320x160 colorful image with a textured road, vegetation, nice scenery.  Our goal will be to focus on what matters, the road and how we have to steer based on the road.  The preprocessing section describes our attempt at simplifying the image data.

Once the problem has been simplified, the appropriate model architecture can be chosen, which is covered in the Model overview section.

##Training Process

The student has the option of recording their own training or using the data provided by Udacity.  I chose to examine the provided data and found that when the left and right cameras are also used, there is sufficient data to proceed with a solution.

Specific extreme recovery training was not necessary as the left and right camera images were used and through fine tuning of the added steering bias produced enough hard-steering (recovery) situations.

##Proprocessing of Images and Data

The diagram below shows the preprocesing steps as well as the data augmentation, which results in a very simple image and enough examples to train most any model.
![Preprocessing of Images and Data](solution approach.png)
The test track is an oval which is driven in a counter clockwise fashion.  This results in all steering angles to be slightly to the left.  Flipping the images not only doubles the amount of training data, but it evenly distributes all steering angles.  So the model should be able to drive clockwise on this same track.

Since standard python data structures were used and the stored images were 32x32x1, they all fit into memory nicely and no generators were needed.

The training data was shuffled and split into a 17.5% validation.  Testing will occur solely on the virtual test track.

##Model overview

Since the data we are processing is basically a dark greyscale triangular blob that is either centered, leaning to the left of the right, we can use even a simplified architecture.  Even simpler than the one used in our traffic sign classifier project.  It turns out that only one convolutional layer is enough.

This is a basic model starting with normalization, 2d convolution with rectified linear unit activation which then gets pooled and dropped and flattened.

The model was compiled using the Adam optimizer.

![Model Overview](model overview.png)

###Keras model summary
```python
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 32, 32, 1)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 16)    160         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 7, 7, 16)      0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 7, 7, 16)      0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 784)           0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             785         flatten_1[0][0]                  
====================================================================================================
Total params: 945
Trainable params: 945
Non-trainable params: 0
____________________________________________________________________________________________________
```
#Conclusion
The car was able to navigate all difficult portions of the test track (the branch off into the dirt portion and the tight corner two thirds of the way through) and complete endless laps at the appropriate speed.