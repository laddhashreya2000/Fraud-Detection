# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()#initialize the white area
pcolor(som.distance_map().T)#distance map gives all the mean distances in the som in a matrix(mid matrix) so we take a transpose T
#now we need to know whether white color corresponds to higher or lower mid so we need a legend
colorbar()#this gives a legend, but these are normalized values of the mid
#now we can add markers which customers got approval or didn't get approval. Red circles = who didn't get approval, green squares = who got approval
markers = ['o', 's']
colors = ['r', 'g']
#now we loop over all the customers and for each customer we get the winning node, and color it red or green 
for i, x in enumerate(X):
    w = som.winner(x)#x is the vector for different customers. It will give winning node. Now plot marker on the node.
    plot(w[0] + 0.5,#to put marker at the center of the square w[0] is the x coordinate of the winning node.
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)#mapping from winning node to customers.
frauds = np.concatenate((mappings[(2,5)], mappings[(7,3)]), axis = 0)#the coordinates of the winning node, concatenate vertically.
frauds = sc.inverse_transform(frauds)


customers = dataset.iloc[:,1:].values
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if(dataset.iloc[i,0] in frauds):
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Part 2 - Now let's make the ANN!

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
# classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#we can also use shuffle and verbose parameters of complie
# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)#since few training data and few features

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(customers)
#now sort the array in python....include custid first
y_pred = np.concatenate((dataset.iloc[:,0:1], y_pred), axis = 1)

y_pred = y_pred[y_pred[:,1].argsort()]
