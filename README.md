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
