#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ssl
import csv
import copy
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score, f1_score, roc_curve 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import scikitplot as skplt
from scipy.special import softmax

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D,MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist


ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


def data_preprocessing():
    
    df = pd.read_csv("train.csv")
    df = df.values
    X = df[:,1:]
    Y = df[:,0]
    
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    return X_train, X_validation, Y_train, Y_validation


# In[ ]:


X_train, X_validation, Y_train, Y_validation = data_preprocessing()

classifier = LinearSVC(penalty="l2",loss="squared_hinge",tol=1,random_state=0,max_iter=9000,C = .8)

classifier.fit(X_train, Y_train)
predicted = classifier.predict(X_train)


# In[ ]:


print(predicted)


# In[ ]:


print(accuracy_score(predicted,Y_train))


# In[ ]:


print(confusion_matrix(predicted,Y_train))


# In[ ]:


val_predicted = classifier.predict(X_validation)
print(accuracy_score(val_predicted,Y_validation))
print(confusion_matrix(val_predicted,Y_validation))


# In[ ]:



X_train, X_validation, Y_train, Y_validation = data_preprocessing()

gnb = GaussianNB()
gnb.fit(X_train, Y_train) 

train_predicted = gnb.predict(X_train)
validation_predicted = gnb.predict(X_validation) 

print("train accuracy: ",accuracy_score(train_predicted,Y_train))
print(confusion_matrix(train_predicted,Y_train))
print("Validation accuracy: ",accuracy_score(validation_predicted,Y_validation))
print(confusion_matrix(validation_predicted,Y_validation)) 


# In[ ]:


X_train, X_validation, Y_train, Y_validation = data_preprocessing()

X_train = X_train/255
X_validation = X_validation/255

clf = SVC(kernel='poly',degree=3)
clf.fit(X_train, Y_train)

train_predicted = clf.predict(X_train)
validation_predicted = clf.predict(X_validation) 

print("train accuracy: ",accuracy_score(train_predicted,Y_train))
print(confusion_matrix(train_predicted,Y_train))
print("Validation accuracy: ",accuracy_score(validation_predicted,Y_validation))
print(confusion_matrix(validation_predicted,Y_validation))


# In[ ]:


test_df = pd.read_csv("test.csv")
X_test = test_df.values
X_test = X_test/255
test_predicted = clf.predict(X_test)
np.savetxt("submission.csv", test_predicted, delimiter=",")


# ## CNN For Digit classification

# In[3]:


global X_train, y_train, X_test, y_test
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[4]:


plt.imshow(X_train[0])


# In[5]:


print(X_train.shape)
print(X_test.shape)


# In[6]:


X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[7]:


print(X_train.shape)
print(X_test.shape)


# In[8]:


x_train, x_validation, Y_train, Y_validation = data_preprocessing()
print(x_train.shape)
print(x_validation.shape)


# In[9]:


x_train = x_train.reshape(33600,28,28,1)
x_validation = x_validation.reshape(8400,28,28,1)
x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_train /= 255
x_validation /= 255


# In[10]:


print(x_train.shape)
print(x_validation.shape)


# In[11]:


print(y_train.shape)
print(y_test.shape)
print(Y_validation.shape)


# In[12]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
Y_validation = to_categorical(Y_validation)


# In[13]:


# print(y_train)
# print(y_test)
# print(Y_validation)


# In[20]:



# model = Sequential()
# model.add(Conv2D(128, kernel_size=(5, 5),activation='relu',input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

input_shape = (28,28,1)
num_category = 10
##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
#32 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.15))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.3))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(num_category, activation='softmax'))


#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 10
#model training
model_log = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 


# In[21]:


test_df = pd.read_csv("test.csv")
x_test = test_df.values
x_test = x_test.reshape(28000,28,28,1)
x_test = x_test.astype('float32')
x_test = x_test/255
y_predicted = model.predict(x_test)


# In[22]:


print(y_predicted)
y_p = np.argmax(y_predicted, axis=1)
print(y_p)
np.savetxt("cnn_submission.csv", y_p, delimiter=",")
# print(Y_validation)


# In[ ]:


print(Y_validation.shape)
print(y_predicted.shape)
y_predicted = np.argmax(y_predicted, axis=1)
Y_validation = np.argmax(Y_validation, axis=1)
print(list(y_predicted[:1]))
print(list(Y_validation[:1]))
print("Validation accuracy: ",accuracy_score(y_predicted,Y_validation))
print(confusion_matrix(y_predicted,Y_validation))

# print("train accuracy: ",accuracy_score(train_predicted,Y_train))
# print(confusion_matrix(train_predicted,Y_train))
# print("Validation accuracy: ",accuracy_score(validation_predicted,Y_validation))
# print(confusion_matrix(validation_predicted,Y_validation))


# In[ ]:




