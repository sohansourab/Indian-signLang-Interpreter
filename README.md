# Indian Sign Language Interpreter
We're working on an AI model that translates Indian sign language to text on the screen. 
## Overview: 
Sign language is one of the oldest and most natural forms of language for communication, hence we have come up with a real-time method using neural networks for finger spelling-based Indian sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by the camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN.
The AI model aims to bridge the communication gap for the deaf and hard-of-hearing community.
## Usage: 
Our AI model uses a camera to recognize the gestures and generates text in real-time.
More than 60 lakh people in India use sign language to communicate. Sign language allows them to learn, work, access services, and be included in the communities.
So, the aim is to develop a user-friendly human-computer interface (HCI) where the computer understands the Indian sign language This Project will help dumb and deaf people by making their lives easy.
## Objective:
 To create a computer software and train a model using CNN that takes an image of hand gestures of Indian sign language and shows the output of the particular sign language in text format and converts it into audio format.
 ## Scope: 
This System will be Beneficial for Both Dumb/Deaf People and the People Who do not understand the Sign Language. They need to do that with sign Language gestures and this system will identify what he/she is trying to say after identification. It gives the output in the form of text-format
## Model architecture:
Our AI uses a convolution neural network (CNN) that works on deep learning.
[![Model Architecture](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/model_arch.png)](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/model_arch.png)
## Training: 
The training of the model is done through various datasets found online. This dataset is divided into training data and testing data. This data is iterated through and filtered by the model and then converted  into a data format that the AI can understand. 
## Evaluation:
The AI is made so that it outputs its progress, accuracy, and loss. We can also use an in-built sub-function called Tensorboard
to display real-time data such as accuracy and loss for both the training data and the testing data and gives a graphical representation for the same over a number of trials of the model.
### EPOCH_ACCURACY:
[![epoch_accuracy](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/epoch_accuracy.svg)](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/epoch_accuracy.svg)
### EPOCH_LOSS:
[![epoch_loss](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/epoch_loss.svg)](https://github.com/sohansourab/ISL-Interpreter/blob/main/Images/epoch_loss.svg)

🟠 - Training data
🔵 - Testing data

## Restrictions:
Given the time constraint and the lack of fast internet and the enoromous file sizes required to be downloaded as the datasets, we were unable to train the model to be more accurate. The AI model is able to recognize letters but not very often and not very accurately.
