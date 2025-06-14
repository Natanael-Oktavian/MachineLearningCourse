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


Logistic regression models are trained using the same process as linear regression models, with two key distinctions:

Logistic regression models use Log Loss as the loss function instead of squared loss.
Applying regularization is critical to prevent overfitting.

Overfitting : Creating a model that matches the training data so closely that the model fails to make correct predictions on new data.
Regularization can reduce overfitting. Training on a large and diverse training set can also reduce overfitting.

![image](https://github.com/user-attachments/assets/88694350-5751-4dbe-ac34-fc03e3589bab)

Regularization, a mechanism for penalizing model complexity during training, is extremely important in logistic regression modeling.
Consequently, most logistic regression models use one of the following two strategies to decrease model complexity:
L2 regularization
Early stopping: Limiting the number of training steps to halt training while loss is still decreasing.


Classification

Confusion matrix : True positive (TP), True negative (TN), False positive (FP), False negative (FN)
Classification metrics :

**Accuracy**

![image](https://github.com/user-attachments/assets/ad46a55b-51af-472e-8fa2-09445edb2623)

Use as a rough indicator of model training progress/convergence for balanced datasets.

**Recall / True positive rate (TPR)**

![image](https://github.com/user-attachments/assets/89480aed-b522-49e6-8a0d-5f09265a4f81)

Decrease the threshold resulted in higher TPR 
Use when false negatives are more expensive than false positives.
Example : Diseases detection

**False positive rate (FPR)**

![image](https://github.com/user-attachments/assets/60f20a52-5fad-49a5-ac33-29a617b5c456)

Increase the threshold resulted in better FPR 
Use when false positives are more expensive than false negatives.
Example : Spam detector

**Precision**

![image](https://github.com/user-attachments/assets/41c25444-c44b-4d02-9d3d-8dbcae2b3fd5)

Use when it's very important for positive predictions to be accurate.
Example : Killer Robot?


Receiver-operating characteristic curve (ROC)

The ROC curve is a visual representation of model performance across all thresholds.

Area under the curve (AUC)

The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative.

![image](https://github.com/user-attachments/assets/c7cc9aba-61f5-4ff2-8486-40fa7f0961f5)



**CODING EXERCISE**

https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/binary_classification_rice.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=binary_classification

**DATA**

- Numerical Data
- Categorical Data

Feature Engineering :

1. Normalization: Converting numerical values into a standard range.
2. Binning (also referred to as bucketing): Converting numerical values into buckets of ranges.

Normalization

The goal of normalization is to transform features to be on a similar scale

normalization methods:

**LinearScaling**

![image](https://github.com/user-attachments/assets/c7453ac2-e1df-4653-8eb5-9133d940fb03)

Linear scaling is a good choice when all of the following conditions are met:

The lower and upper bounds of your data don't change much over time.
The feature contains few or no outliers, and those outliers aren't extreme.
The feature is approximately uniformly distributed across its range. That is, a histogram would show roughly even bars for most values.
Suppose human age is a feature. Linear scaling is a good normalization technique for age because:

The approximate lower and upper bounds are 0 to 100.
age contains a relatively small percentage of outliers. Only about 0.3% of the population is over 100.
Although certain ages are somewhat better represented than others, a large dataset should contain sufficient examples of all ages.

**Z-score scaling**

A Z-score is the number of standard deviations a value is from the mean.
![image](https://github.com/user-attachments/assets/126b1703-6f72-40ab-8fde-5a9fa5817f18)

Z-score is a good choice when the data follows a normal distribution or a distribution somewhat like a normal distribution.


**LogScaling**

![image](https://github.com/user-attachments/assets/40f41dc6-71e9-436f-b068-f813c1b90360)

Log scaling is helpful when the data conforms to a power law distribution. Casually speaking, a power law distribution looks as follows:

Low values of X have very high values of Y.
As the values of X increase, the values of Y quickly decrease. Consequently, high values of X have very low values of Y.

**Clipping**

Clipping is a technique to minimize the influence of extreme outliers. In brief, clipping usually caps (reduces) the value of outliers to a specific maximum value. 

![image](https://github.com/user-attachments/assets/c132c06e-557e-474a-8ccb-9348451400e0)

**Binning**

![image](https://github.com/user-attachments/assets/d9ee369e-4cbf-4d73-8fff-e394aff6b267)

Binning is a good alternative to scaling or clipping when either of the following conditions is met:

The overall linear relationship between the feature and the label is weak or nonexistent.
When the feature values are clustered.

**Scrubbing**

![image](https://github.com/user-attachments/assets/0d9e8984-369a-4ba3-ba20-355ac78ffa1f)

Once detected, you typically "fix" examples that contain bad features or bad labels by removing them from the dataset or imputing their values.


Best practices for working with numerical data:

Remember that your ML model interacts with the data in the feature vector, not the data in the dataset.

Normalize most numerical features.

If your first normalization strategy doesn't succeed, consider a different way to normalize your data.

Binning, also referred to as bucketing, is sometimes better than normalizing.

Considering what your data should look like, write verification tests to validate those expectations. For example:
The absolute value of latitude should never exceed 90. You can write a test to check if a latitude value greater than 90 appears in your data.
If your data is restricted to the state of Florida, you can write tests to check that the latitudes fall between 24 through 31, inclusive.

Visualize your data with scatter plots and histograms. Look for anomalies.

Gather statistics not only on the entire dataset but also on smaller subsets of the dataset. That's because aggregate statistics sometimes obscure problems in smaller sections of a dataset.

Document all your data transformations.


**Categorical Data**

Encoding means converting categorical or other data to numerical vectors that a model can train on. 

![image](https://github.com/user-attachments/assets/9a28a46e-3274-42a8-b94c-1be599158193)

![image](https://github.com/user-attachments/assets/4555f656-e8b6-4656-b3dd-83abfdbc943c)

![image](https://github.com/user-attachments/assets/a33de071-6346-4390-927b-bb3f6010b08c)

![image](https://github.com/user-attachments/assets/957a8dea-ed79-46b2-8ea8-eaab17fb23eb)



**Label**
Direct versus proxy labels
Consider two different kinds of labels:

Direct labels, which are labels identical to the prediction your model is trying to make. That is, the prediction your model is trying to make is exactly present as a column in your dataset. For example, a column named bicycle owner would be a direct label for a binary classification model that predicts whether or not a person owns a bicycle.
Proxy labels, which are labels that are similar—but not identical—to the prediction your model is trying to make. For example, a person subscribing to Bicycle Bizarre magazine probably—but not definitely—owns a bicycle.

**Imbalanced datasets**


One way to handle an imbalanced dataset is to downsample and upweight the majority class. Here are the definitions of those two new terms:

Downsampling (in this context) means training on a disproportionately low subset of the majority class examples.
Upweighting means adding an example weight to the downsampled class equal to the factor by which you downsampled.

Step 1: Downsample the majority class. Consider the virus dataset shown in Figure 5 that has a ratio of 1 positive label for every 200 negative labels. Downsampling by a factor of 10 improves the balance to 1 positive to 20 negatives (5%). Although the resulting training set is still moderately imbalanced, the proportion of positives to negatives is much better than the original extremely imbalanced proportion (0.5%).

Step 2: Upweight the downsampled class: Add example weights to the downsampled class. After downsampling by a factor of 10, the example weight should be 10. (Yes, this might seem counterintuitive, but we'll explain why later on.)

You might also be wondering whether upweighting cancels out downsampling. Yes, to some degree. However, the combination of upweighting and downsampling enables mini-batches to contain enough minority classes to train an effective model.

Upweighting the minority class by itself is usually easier to implement than downsampling and upweighting the majority class. However, upweighting the minority class tends to increase prediction bias.

Downsampling the majority class brings the following benefits:

Faster convergence: During training, the model sees the minority class more often, which helps the model converge faster.
Less disk space: By consolidating the majority class into fewer examples with larger weights, the model uses less disk space storing those weights. This savings allows more disk space for the minority class, so the model can collect a greater number and a wider range of examples from that class.
Unfortunately, you must usually downsample the majority class manually, which can be time consuming during training experiments, particularly for very large datasets.

Rebalance ratios
How much should you downsample and upweight to rebalance your dataset? To determine the answer, you should experiment with the rebalancing ratio, just as you would experiment with other hyperparameters. That said, the answer ultimately depends on the following factors:

The batch size
The imbalance ratio
The number of examples in the training set
Ideally, each batch should contain multiple minority class examples. Batches that don't contain sufficient minority classes will train very poorly. The batch size should be several times greater than the imbalance ratio. For example, if the imbalance ratio is 100:1, then the batch size should be at least 500.

**Dividing the original dataset**
Training, validation, and test sets

**Overfitting**

Overfitting means creating a model that matches (memorizes) the training set so closely that the model fails to make correct predictions on new data. An overfit model is analogous to an invention that performs well in the lab but is worthless in the real world.

Generalization is the opposite of overfitting. That is, a model that generalizes well makes good predictions on new data. Your goal is to create a model that generalizes well to new data.

Detecting overfitting
The following curves help you detect overfitting:

loss curves
generalization curves
A loss curve plots a model's loss against the number of training iterations. A graph that shows two or more loss curves is called a generalization curve. The following generalization curve shows two loss curves:

![image](https://github.com/user-attachments/assets/7cf84803-b3a1-4313-b8c7-40440a203210)

What causes overfitting?
Very broadly speaking, overfitting is caused by one or both of the following problems:

The training set doesn't adequately represent real life data (or the validation set or test set).
The model is too complex.

Generalization conditions

A model trains on a training set, but the real test of a model's worth is how well it makes predictions on new examples, particularly on real-world data. While developing a model, your test set serves as a proxy for real-world data. Training a model that generalizes well implies the following dataset conditions:

- Examples must be independently and identically distributed, which is a fancy way of saying that your examples can't influence each other.
- The dataset is stationary, meaning the dataset doesn't change significantly over time.
- The dataset partitions have the same distribution. That is, the examples in the training set are statistically similar to the examples in the validation set, test set, and real-world data.

![image](https://github.com/user-attachments/assets/b3301da6-3a5e-412a-bfec-cfed82ddeb14)

![image](https://github.com/user-attachments/assets/167d88e6-de99-4976-ac18-cbe8d7f5d2d0)

![image](https://github.com/user-attachments/assets/4266efba-2e36-4388-a249-d750eae459c0)





