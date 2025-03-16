# Comment Classification Project

## Project Overview

The **Comment Classification Project** focuses on the task of categorizing textual comments into predefined categories. This task is crucial in fields such as social media moderation, customer feedback analysis, and sentiment analysis. The main objective of this project is to develop a machine learning model capable of classifying comments into different categories based on their content.

This project involves the use of deep learning techniques with **TensorFlow** in **Google Colab** to build and train the model. The dataset used for training and testing contains labeled comments that are categorized based on their sentiment or content.

## Dataset

The dataset used in this project is provided by Kaggle through the **Jigsaw Toxic Comment Classification Challenge**. The dataset consists of comments that need to be classified into categories such as toxic, non-toxic, and others. You can access the dataset and the rules for participation by following the link below:

[Dataset Link ](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

## Approach

The project uses **TensorFlow** and a **Sequential neural network** architecture with LSTM layers to process the comments. Preprocessing includes tokenization, padding, and vectorization to convert the raw text data into a format suitable for the model.

The model is trained using a **cross-entropy loss function** and optimized with the **Adam optimizer**.

## Goals

- To classify comments into multiple categories such as positive, negative, and neutral.
- To enhance the model's performance with possible improvements, including model architecture, hyperparameter tuning, and data augmentation.
