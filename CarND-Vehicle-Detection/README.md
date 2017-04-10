**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image4]: ./output_images/boxes_with_heatmap_2.png
[image5]: ./output_images/boxes_with_heatmap.png
[image6]: ./output_images/labelled_heatmap.png
[image7]: ./output_images/two_blobs.png
[video1]: ./output_images/project_video_processed.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it.  Since it came half-way filled out, I will retain most of it and simply add relevant remarks.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the [IPython](Vehicle Detection.ipynb) notebook.  Its a achieved through the skimage.feature.hog function.

I reused the code for training from the code we built during the classwork.  All I did was point it towards the training data:
`cars = glob.iglob('vehicles/**/*.png')
notcars = glob.iglob('non-vehicles/**/*.png')`

It ran rather fast, but consumed roughly 7.6 gigabytes of memory.

`58.69 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
20.36 Seconds to train SVC...
Test Accuracy of SVC =  0.9916
My SVC predicts:  [ 0.  1.  1.  1.  1.  0.  1.  0.  1.  0.]
For these 10 labels:  [ 0.  1.  1.  1.  1.  0.  1.  0.  1.  0.]
0.001 Seconds to predict 10 labels with SVC`

#### 2. Explain how you settled on your final choice of HOG parameters.

I was satisfied with the accuracy of the HOG parameters presented in the coursework.  Thus the resulting values were:
`colorspace = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the sliding window subsampling hog approach described in the coursework.  There are 2 cells per step (8 pixels_per_cell * 2 cells_per_step = 16 pixels)   The large overlap can be seen in the test image below.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  HOG features were extracted for the entire region of interest at once (ctrans_tosearch), then the results were examined one 64x64 pixel window at a time.  Here is the result:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]

There are some brief false positives, but it is relatively smooth.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

#### Here is a frame and its heatmap, showing multiple boxes:

![alt text][image5]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from bounding boxes:
![alt text][image6]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from bounding boxes where two cars are seperated:
![alt text][image7]


The Weighted heatmap basically averages the heatmap from previous frames which helps generate a smoother box with less false positives.

See `cv2.addWeighted(process.avg_heat, 0.8, heat, 0.2, 0.)`, where heat is the current thresholded heatmap and avg_heat is the running average.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The single frame detection worked well for all examples, and was reasonable stable on the video.  I used the most simple form of temporal smoothing, which was treating the heatmap image as a canvas with a short memory.

To make classification more robust I would not use HOG features, but a deep learning based approach.   This would then be trained for pedestrians, two wheelers, trucks and other common objects (construction zones).
