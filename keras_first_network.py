# -*- coding: utf-8 -*-
"""
       (`-()_.-=-.
       /66  ,  ,  \
     =(o_/=//_(   /======`
         ~"` ~"~~`        C.E.
         
Created on Thu Jan  7 09:14:05 2021
@author: Chris
Contact :
    Christopher.eaby@gmail.com
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
import os

#loading the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter = ',')

#creating input and output variables
x = dataset[:,0:8]
y = dataset[:,8]

#creating the model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compiling the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#fit the model into the dataset
model.fit(x, y, epochs = 150, batch_size = 2)


#evaluation of model accuracy
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' %(accuracy*100))


#probability predictions
predictions = model.predict_classes(x)

#summarize first 5 classes
for i in range(3):
    print('%s => %d (expected %d)' %(x[i].tolist(), predictions[i], y[i]))     

#turn model into json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
#weights to HDF5
model.save_weights("model.h5") 
print('Saved model from disk')      

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model = model_from_json(loaded_model_json)
print('Loaded model from disk')

#evaluate loaded model on test data
loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
score = loaded_model.evaluate(x, y, verbose = 0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)
print(model.summary())


