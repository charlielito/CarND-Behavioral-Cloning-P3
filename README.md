# **Behavioral Cloning**
Carlos Andres Alvarez
Self driving nanodegree Udacity

## Writeup

---


[//]: # (Image References)

[image1]: ./results/nvidia_net.png "Model Visualization"
[image2]: ./results/video.gif "Video"
[image3]: ./results/losses.png "Loss graph"
[image4]: ./results/mae.png "MAE graph"
[image5]: ./results/left.jpg "Left Image"
[image6]: ./results/center.jpg "Normal Image"
[image7]: ./results/right.jpg "Right Image"
[image8]: ./results/cropped.png "Cropped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---
### Files Submitted & Code Quality

#### 1. Files in project

My project includes the following files:
* train.py containing the script to create and train the model
* model.py that defines different conv architectures
* data_augmentation.py for image data augmentation (actually dont used in the final version)
* drive_estimator.py for driving the car in autonomous mode using the Estimator API from tensorflow
* export/nvidia_dataset/1552541693/ containing a trained convolutional neural network as a saved model
* writeup_report.md summarizing the results

#### 2. Functional code
Using the Udacity provided simulator and my drive_estimator.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py export/nvidia_dataset/1552541693 22
```
where 22 is the desired velocity set point.

#### 3. Usable and readable code

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

I didn't use the keras API since tf.estimator shows a more flexible and configurable mode to define the network and run the training. In the model.py is the architecture defined.

### Model Architecture and Training Strategy

#### 1. Model architecture

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 36 and 64. See `nvidia_net` function in model.py line 416.The total architecture can be seen as follows:

![alt text][image1]

This network is based on Nvidia's paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf). The only changes are the image input dimensions and the fully connected layers (number of neurons). All activations are RELU. I used also dropout after each layer with a drop rate of 0.4 for convolutional layers and 0.5 for dense layers. Also batch normalization is applied to all layers including the input layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers, as said, in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Here I create 2 tf.estimator Specs for evaluating in different datasets. The split was made 80-20 from the dataset.

```
# Specs
train_spec = tf.estimator.TrainSpec(
   input_fn=lambda: create_dataset(filenames, labels, EPOCHS, BATCH_SIZE),
   max_steps=EPOCHS*steps_per_epoch)

eval_spec = tf.estimator.EvalSpec(
   input_fn=lambda: create_dataset(val_filenames, val_labels, EPOCHS, BATCH_SIZE),
   steps=None,
   exporters=[exporter],
   start_delay_secs=10,  # Start evaluating after 10 sec.
   throttle_secs=30  # Evaluate only every 30 sec
)

# Train loop
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 129). I used a learning rate of 0.001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center driving and in training time the model used also right and left camera information.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a similar network to the Nvidia for self driving in the cited paper. If they managed to work in that paper in real life, it should also work for this simulation task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and also low mean squared error on the validation set. This implied that the model was not overfitting. I tried to use deeper networks with more filter but it didn't improve this metric, so I kept going with the original architecture of Nvidia.


The final step was to run the simulator to see how well the car was driving around track one. The results were good at speed lower than 25. For speeds above that the car start to oscillate, but I think is due that the network cannot run as fast as the dynamics requires.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The network also outputs its salient features using the technique proposed in this Nvidia's paper [Explaining How a Deep Neural Network Trained with
End-to-End Learning Steers a Car](https://arxiv.org/pdf/1704.07911.pdf). Basically it tracks in each feature volume in each layer the most activated pixels and deconvolutes it to adjust to the size of the previous layer. All is then average. The result of the autonomy and this visualization can be seen in the next gif:


![alt text][image2]

As seen, the network learned to pay attention to the pixels that correspond to the lane lines or objects. The full video with normal speed (the gif is at x2) can be found [here](https://github.com/charlielito/CarND-Behavioral-Cloning-P3/blob/master/results/video.mp4)

#### 2. Final Model Architecture

The final model architecture (model.py lines 416-582) consisted of a convolution neural already described in a previous section.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I then recorded three laps on same track but going in the opposite direction. I also used the left and right cameras, to capture images for recovering. Thos images look like the following:

![alt text][image5]
![alt text][image7]

The steering label for those images is the steering command from the center cameras -+ a fixed number. That number was chosen to be 0.2 and it seemed to work well.

To augment the data set, I also make some distortions to the image, like blurry filters, rotation, scale, etc, thinking that this could help the model to overfit less the data. The data augmentation pipeline can be seen in the file data_augmentation.py. At the end this augmented data was not used since not major improvements were noticed.

After the collection process, I had around 24k number of data points (no data augmentation). I then preprocessed this data by cropping the images from the top and from the bottom a fixed number of pixels. This, to not see the front part of the car and to focus only in the image from the road (the upper part does not matter to drive). The crop up was set to 50 pixels and the crop down to 30 pixels. The images at the end look like the following

![alt text][image8]


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by the loss and MAE (mean squared error) graphs (see below). The mae and loss kept going down, so I let the model train for that long steps.

![alt text][image3]
![alt text][image4]


