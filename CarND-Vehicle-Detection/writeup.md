##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell number 3 & 4 of the IPython notebook (vehicle_detection). I basically used skimage.feature.hog to extract hog features.

I started by reading in all the `vehicle` and `non-vehicle` images (code cell 3).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like for parameter combinations and then used that to sort of identify the final parameter values I would be using. The code for this can be seen in cell number 303. Here's an example below that simply visualized the hog features for random vehicle & non vehcile images. I used the following hog parameters `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and grayscaled image

![Random Vehicle and Non-Vehicle Image with hog features](output_images/random.png?raw=true)

My eventual feature extractor is a combination of hog features, color conversion spatial binning featurs & color histogram features concatenated together. The parameters I pass to the `extract_features()` method (in code cell 46) is in code cell 45.

####2. Explain how you settled on your final choice of HOG parameters.

I started trying out various combination of parameters for the feature extractor for extracting hog features, spatial binning & computing color histogram features. My final parameters selected are in code cell 45.
I tested with `orientations=8` and `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and finally the same with `orientations=11` which turned out to be better and hence fixed that as one of the param values.
I selected color space based on training & testing results of classifier & hog feature extractor. I eventually zeroed down on `YCrC`b color space & used all 3 channels to extract hog features to ensure I don't loose features in any of the channels.

Final Hog, Color & Spatial Parameters used.
````
    colorspace = 'YCrCb'
    orient = 11
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'
    spatial_size = (16,16)
    hist_bins = 16
````

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My final feature extractor `extract_features()` is in code cell 46. As mentioned its a combination of hog, color-histogram & spatially-binned features.
In code cell 48 I use the above method to extract car & non features from the data-set provided.
In code cell 49 I first prepare the feature vectors as a numpy array where each row is a feature vector. This is done to get the data in the right format as needed by the normalizer. I normalize the training and test data to sort of prepare it before applying any classifier.
I use the `sklearn.preprocessing.StandardScaler` to standardize the features by removing mean and unit variance.
I prepare the labels vector as well and finally randomly shuffle and split the data into training and test sets using `train_test_split()` method. This is then used for further for the actual classification (training & detection step). All of the above is in code cells 48 & 49.

In Code cell 50, I actually train the classifier using `sklean.svm.LinearSVC`. It has a linear kernel. It uses training data prepared in the step above.
Once the classifier was trained I tested its accuracy using `svc.score` on the test set. The obtained accuracy is **98.79%**. I then save the classifier to the pickle file for future use if needed.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I wrote a function called `find_cars()` based on the lectures that computes the hog features, color & spatial features and then makes predictions. It then uses the scales, cells_per_block and window size to define bounding box for each positive detection. The function is available in code cell 261.

Instead of a sliding window approach I sub-sampled hog features to get all overlaying windows. I initally tested with a single scale of 1.5, but noticed it did not detect cars on all images. Hence I went with a multi-scale approach and experimented with random scales before zeroing on 1.0, 1.5 & 1.7 as my 3 scales. I tried higher scale values of 2.0, 3.5 etc. and greater realized that with these y_ranges it was not detecting all the cars. The current ones chosen detect multiple overlapping boxes, but its better than missing out on information. I then played with different Y_range values to match up with these scales to finally create a pipeline that uses 5 different y_ranges along with these scales. I experimented with `cells_per_block=1` which was causing many false positives and overlaps and hence setlled on `cells_per_block=2`which gives the right amount of overlap.

In the `find_cars()` function I used a window size of 64. I experimented with random higher values but settled on 64 since it was giving enough information and I din't want to increase the size of the bounding box further.

Code cell 222 has the `draw_boxes()` method that is used to draw the bounding boxes given to it.
Code cell 306 has code to test/visualize the various scales/Y-ranges on a sample image and then eventually displays the combined multiscale image.
This is a sample of how the multiscale approach looks for `test_images/test4.jpg`.

![Multi Scaled Windows with Each Step](output_images/find_cars.png?raw=true)



### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/htPAVNBgszU)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Code cell 37 has some functions that are basically used for building heat maps, thresholding heat maps and finally drawing labelled cars.
I basically wrote a function `add_heat()` that adds +1 for all pixels within the windows where a positive detection is reported by the classifier. This heatmap is then thresholded to return only hot parts of the image minus the false positives(where a vehicle is detected). We then apply the `scipy.ndimage.measurements.labels()` function to count all features in an image. It sort of groups nonzero pixels belonging to a car togther to return tuple with no of cars detected. I then invoke `draw_labeled_bboxes()` to render the labeled boxes on to the copy of the image. The function basically iterate through the labels array and pixels with each car_number label value. We then form bouding box around nonzerox and nonzeroy values using min and max values.

I basically experimented with each of these steps. The code for this is in code blocks 311, 314, 310. Some examples of each step:
**Without Thresholding**
![Without Thresholding HeatMap](output_images/without_thresholding.png?raw=true)

**With Thresholding**
![With Thresholding HeatMap](output_images/with_thresholding.png?raw=true)

**With boxes drawn on image**
![With Thresholding HeatMap](output_images/final_withboxes.png?raw=true)

Now for the video I maintain the bounding box detections for the last 20 frames. I then create a heatmap using all of the detections of the last 20 frames, and then apply a dynamic threshold (threshold of 20 + len(bounding_boxes)//2) on the image using all these detections.
The core steps are the same as described above, used across 20 frames so that there is less jumpiness/wobbly detetction in the video.
The code for this is in code cell 293. The function is `process_frame()`. This is invoked in cell 294 where the video is processed. The class `Detection()` holds the bounding box information for the last 20 frames.

Detetction examples of all the test_images:
![With Thresholding HeatMap](output_images/all_test_images.png?raw=true)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I faced issues with identifying the ideal scales & y range values. I had to spend a lot of time experiementing with values across different frames of images before zeroing on the 3 scale values and yranges. I also wasn't sure about the colorspace to use. I first tried HSV & RGB and I found the classifier accuracy on the lower side so I ended up trying YCrCb which seemed to work perfectly fine. All this experiementation took quite some time.
One issue I have noticed is detections when the vehicle is sort of parallel. its hard to detect vehicle parallel vehicles since its not in the exact frame of view.

I would probably like try to average the frames better & threshold based on lighting condition to improve the detection on low lighting conditions and lanes not divided by the raised median.  
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

