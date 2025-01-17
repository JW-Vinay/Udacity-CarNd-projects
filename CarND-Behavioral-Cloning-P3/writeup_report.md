**Behavioral Cloning**

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

1. Submission includes all required files and can be used to run the simulator in autonomous mode
**Answer** My project includes the following files:
    * model.py containing the script to create and train the model
    * drive.py for driving the car in autonomous mode
    * model_final.h5 containing a trained convolution neural network
    * writeup_report.md summarizing the results

2. Submission includes functional code
**Answer** Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
    ```sh
    python drive.py model_final.h5
3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
##### 1. Design/Approach
* I spent some time looking at the data and at first split it into training & validation set (20%). I then experimented using the center camera images only.I got things working together by simply using the well trusted Lenet Architecture model.
* The next step before doing anything based on the jittery output was to normalize the input. So I went ahead and did that.
* After a few runs I realized there was sort of a left turn bias so I augmented more images by flipping each image horizontally. Also I added random brightness correction to the images.
* Next I used the left & right camera images as part of the training set as well.
* I also added layer of Cropping to the architecture preprocessing to crop out useless portions of the image.
* All of these steps did work fine for some portions of the track with appropriate dropout & maxpooling layers.
* Eventually I switched to the nvidia neural network architecture described in the paper linked below. [Nvidia Paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* The architecture is described in detail later below.
##### 2. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting **[lines 153 - 155]**. It was run for 5 epochs only.The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. Data was randomly shuffled before splitting into training and validation sets.
Dropout was not used to stay true to the Nvidia model. Also initial uses of Dropout didn't really help with the training. I eventually used l2 regularization to avoid overfitting by penalizing all the weights in the fully connected layers and convolution layers. I used a beta value  of 0.001 (line 118-133). Initially I started with a beta value of 0.01, but switched to 0.001 since it improved the results.
Maxpooling was not used because it tends to make output slighty invariant to input change which is not really needed here when attempting to center the car to the middle of the lane.
I also generalized the model by generating more training data by driving the car in the clockwise direction on the track so it is not biased to any one direction.
This was in a separate csv file 'dl2.csv' **[line 33]**. This was read and added to the training set.
The validation set helped determine if the model was over or under fitting. The ideal number of epochs was in the range of 5-7 (I went with 5) as evidenced by the final output video final_run.mp4. For any values greater than 7 epochs I could see the MSE increasing on the validation set.
I initially used 10 epochs and a different parameter value set but changed things around when I fixed an unforseen bug about the initial color model of the images. Once this bug was fixed convergence was in the 5-7 epochs range. Also this model works way better on the challenge track (almost completing it). Proving that it was not overfitted. I will probably tweak it later to complete the challenge track as well.
##### 3. Model parameter tuning
The model used an adam optimizer, so manual tuning was not necessary. I started with initial learning rate of 0.001 but then switched to 0.0001 since it seemed to further reduce validation loss and perform flawlessly on track 1 and pretty well on track 2 as well.(model.py line 158).
##### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I used a combination of lane driving in the counter clock wise & clock direction to generalize the model. I then preprocessed the collected training data by adding horizontally flipped version of the images (left, right & center camera) to deal with left biases. Before passing it through the model in batches the data was shuffled randomly.

Example of images:
1. Example 1 (Original, Random Brightness Adjusted, Flipped Image)
   * ![Original Image](augmented/1495213584.jpg "Original Image")
   * ![Brightness Adjusted Image](augmented/1495213584_brightened.jpg "Brightness Adjusted Image")
   * ![Flipped Image](augmented/1495213584_flipped.jpg "Flipped Image")

2. Example 2 (Original, Random Brightness Adjusted, Flipped Image)
   * ![Original Image](augmented/1495213585.jpg "Original Image")
   * ![Brightness Adjusted Image](augmented/1495213585_brightened.jpg "Brightness Adjusted Image")
   * ![Flipped Image](augmented/1495213585_flipped.jpg "Flipped Image")

##### 5. Final Model Architecture
* As mentioned previously I used the Nvidia Neural Network model archictecture. It consists of the following layers (lines 114-145)
    1. It has 3 convolutional layers with filter size 5X5. It uses a stride of 2X2 along with valid padding
    2. The output is passed through 2 more convolutional layers with filter size 3X3. The stride used is 1X1 along with valid padding.
    3. Each convolution layer is followed by a Leaky ELU activation function. I initially used RELU but switched to ELU because they have smoother derivates at 0 and should therefore be better at predicting continous values.
    4. The outputs are flattened next before passing through the fully connected layers.
    5. This output is then passed through 3 fully connected layers followed by the final logit (also a fc layer) which results in the output (steering angle).
    6. Each Fully connected layer (except the logit) is followed by an ELU activation function.
* I used Keras to implement this sequential model. Hence I normalized the images in the model using a lambda layer. I also cropped each image as it passed through the models to remove the top 60 pixels and bottom 20 pixels.

Here is a visualization of the nvidia architecture

![Nvidia architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

There are also some older models in the previous_models folder. particularly m2_final.h5 which has slightly fewer epochs but performs equally well on Track 1 but slightly better on Track 2.
