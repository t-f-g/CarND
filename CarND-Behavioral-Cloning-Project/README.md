#Project Overview

This project will analyze the output of a driving simulation and predict the appropriate steering angle.  The overall flow will look like the following:
![Overview](project overview.png)

##Proprocessing of Images and Data

The first goal is to simplify the data so that the model can also be simple.  Then we need enough data so that it can generalize.
![Preprocessing of Images and Data](solution approach.png)

Since standard python data structures were used and the stored images were 32x32x1, they all fit into memory nicely and no generators were needed.

The training data was shuffled and split into a 17.5% validation.  Testing will occur solely on the virtual test track.

##Model overview

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