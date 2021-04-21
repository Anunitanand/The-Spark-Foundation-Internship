# The-Spark-Foundation-Internship

This repository contains the tasks that I completed while working as an intern for [The Sparks Foundation](https://www.thesparksfoundationsingapore.org/).

- I build a deep neural network model to classify traffic signs present in the image into different categories.
- The dataset I have used to train my traffic sign classifier is the German Traffic Sign Recognition Benchmark (GTSRB).
- This GTSRB dataset consists of 43 traffic sign classes and nearly 50,000 images.

## Model Accuracy is up to 96%.

My approach to building this traffic sign classification model is discussed in four steps:

- Explore the dataset
- Build a CNN model
- Train and validate the model
- Test the model with the test dataset


Now coming to our first step

## Step 1: Explore the dataset
Our ‘train’ folder contains 43 folders each representing a different class. The range of the folder is from 0 to 42. With the help of the OS module, we iterate over all the classes and append images and their respective labels in the data and labels list.
The PIL library is used to open image content into an array.

Finally, we have stored all the images and their labels into lists (data and labels).
We need to convert the list into numpy arrays for feeding to the model.
The shape of data is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value).
With the sklearn package, we use the train_test_split() method to split training and testing data.
From the keras.utils package, we use to_categorical method to convert the labels present in y_train and t_test into one-hot encoding.

## Step 2: Build a CNN model
To classify the images into their respective categories, we will build a CNN model (Convolutional Neural Network). CNN is best for image classification purposes.

Then, We compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.


## Steps 3: Train and validate the model
After building the model architecture, we then train the model using model.fit(). I tried with batch size 32 and 64. Our model performed better with 64 batch size. And after 15 epochs the accuracy was stable.

Our model got a 95% accuracy on the training dataset. With matplotlib, we plot the graph for accuracy and the loss.

Now our final step

Our dataset contains a test folder and in a test.csv file, we have the details related to the image path and their respective class labels. We extract the image path and labels using pandas. Then to predict the model, we have to resize our images to 30×30 pixels and make a numpy array containing all image data. From the sklearn.metrics, we imported the accuracy_score and observed how our model predicted the actual labels. We achieved a 96% accuracy in this model.

