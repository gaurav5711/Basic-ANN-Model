import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values # Deciding which values can affect the result
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_1 = LabelEncoder()
x[:,1] = label_encoder_1.fit_transform(x[:,1])
x[:,2] = label_encoder_1.fit_transform(x[:,2])
one_hot_encoder = OneHotEncoder(categorical_features=[1])
x = one_hot_encoder.fit_transform(x).toarray()
#Avoiding dummy variable trap
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import keras
from keras.models import Sequential # Used to initialise ANN
from keras.layers import Dense # Used to build layers

#Initialising ANNs can be done by defining it as a sequence of layers or as a sequence of graphs
classifier = Sequential() # classifier is ANN

# Adding input layer and first hidden layer

# will choose rectifier function for hidden layer and will choose sigmoid function for output layer 
# Rectifier function --> Best activation function proven mathematically 
# Sigmoid function --> Applicable here as it is a classifier problem

#units = no. of nodes you want to add in the hidden layer 
#As a tip, dim of hidden layer = average of dim of input and output layer............. better approach is to use parameter pruning (Section 10)
#kernel_initializer="uniform"--> initialises using a uniform function and makes sure that the values are closed to zero
#activation="relu" --> rectifier function
#input_dim=11 --> needs to provide only to the first hidden layer created as it doesn't know about the input layer. From next time, it will know what
#to expect from previous layer. 11 = no. of independent variables in input
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))

#Adding second hidden layer
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))

#Adding second output layer
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

#Compiling the ANN
#adam=efficient stochastic gradient descent
#metrics=... -> expects list of metrics. We will continuously get the value accuracy which will tell us about the accuracy of the model
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Making the prediction and evaluating the model
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#Predicting the test results
y_pred = classifier.predict(x_test)
y_pred = y_pred > 0.5

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print cm