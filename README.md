# MachineLearningCourse
Summarize of google machine learning crash course (https://developers.google.com/machine-learning)

ML is the process of training a piece of software, called a model, to make useful predictions or generate content (like text, images, audio, or video) from data.
A model in ML is a mathematical relationship derived from data that an ML system uses to make predictions.

Types of ML Systems
- Supervised learning
- Unsupervised learning
- Reinforcement learning
- Generative AI


Supervised learning 

Supervised learning models can make predictions after seeing lots of data with the correct answers and then discovering the connections between the elements in the data that produce the correct answers
Two of the most common use cases for supervised learning are regression and classification.
Regression model predict numeric value. 
Classification models predict the likelihood that something belongs to a category. 
Classification models are divided into two groups: binary classification and multiclass classification. 


Unsupervised learning

Unsupervised learning models make predictions by being given data that does not contain any correct answers. An unsupervised learning model's goal is to identify meaningful patterns among the data. In other words, the model has no hints on how to categorize each piece of data, but instead it must infer its own rules.
A commonly used unsupervised learning model employs a technique called clustering
Clustering differs from classification because the categories aren't defined by you. For example, an unsupervised model might cluster a weather dataset based on temperature, revealing segmentations that define the seasons. You might then attempt to name those clusters based on your understanding of the dataset.


Reinforcement learning

Reinforcement learning models make predictions by getting rewards or penalties based on actions performed within an environment. A reinforcement learning system generates a policy that defines the best strategy for getting the most rewards.
Reinforcement learning is used to train robots to perform tasks, like walking around a room, and software programs like AlphaGo to play the game of Go.


Generative AI

Generative AI is a class of models that creates content from user input. For example, generative AI can create unique images, music compositions, and jokes; it can summarize articles, explain how to perform a task, or edit a photo.



Supervised Learning

Supervised machine learning is based on the following core concepts:
1. Data
2. Model
3. Training
4. Evaluating
5. Inference


Data

Datasets are made up of individual examples that contain features and a label. 
Features are the values that a supervised model uses to predict the label. The label is the "answer," or the value we want the model to predict.


Model

In supervised learning, a model is the complex collection of numbers that define the mathematical relationship from specific input feature patterns to specific output label values. The model discovers these patterns through training.

Training

A model needs to be trained to learn the mathematical relationship between the features and the label in a dataset.


Linear regression

Linear regression is a statistical technique used to find the relationship between variables. In an ML context, linear regression finds the relationship between features and a label.
![image](https://github.com/user-attachments/assets/6495a967-a718-4ff1-8c78-4d59b5176b87)


y' : is the predicted label—the output.
b : is the bias of the model. Bias is the same concept as the y-intercept
w : is the weight of the feature. Weight is the same concept as the slope in the algebraic equation for a line. Weight is a parameter of the model and is calculated during training.
x : is a feature—the input.


Loss is a numerical metric that describes how wrong a model's predictions are. Loss measures the distance between the model's predictions and the actual labels.
Types of loss
![image](https://github.com/user-attachments/assets/1b015114-c02f-46a7-a827-b3aa7d258c56)

MSE. The model is closer to the outliers but further away from most of the other data points.
MAE. The model is further away from the outliers but closer to most of the other data points.

Gradient descent

Gradient descent is a mathematical technique that iteratively finds the weights and bias that produce the model with the lowest loss.
When training a model, you'll often look at a loss curve to determine if the model has converged.
The loss functions for linear models always produce a convex surface (letter U). As a result of this property, when a linear regression model converges, we know the model has found the weights and bias that produce the lowest loss.


Hyperparameters are variables that control different aspects of training. Three common hyperparameters are:

Learning rate
Batch size
Epochs

Learning rate

Learning rate is a floating point number you set that influences how quickly the model converges. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges, but instead bounces around the weights and bias that minimize the loss. The goal is to pick a learning rate that's not too high nor too low so that the model converges quickly.

Batch size

Batch size is a hyperparameter that refers to the number of examples the model processes before updating its weights and bias.
Stochastic gradient descent (SGD): Stochastic gradient descent uses only a single example (a batch size of one) per iteration. 
Mini-batch stochastic gradient descent (mini-batch SGD): Mini-batch stochastic gradient descent is a compromise between full-batch and SGD. For N 
number of data points, the batch size can be any number greater than 1 and less than N

Epochs

During training, an epoch means that the model has processed every example in the training set once. For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch.


Logistic Regression

Calculating a probability
Sigmoid function
![image](https://github.com/user-attachments/assets/5d234b8c-2234-4a8b-9c9c-fe703302bfe1)

![image](https://github.com/user-attachments/assets/88bdcd53-6848-4516-9509-27c0477b7748)
![image](https://github.com/user-attachments/assets/727dc9b5-465f-4655-9dcc-410d8d25ea00)



