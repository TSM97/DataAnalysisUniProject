# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean
from numpy import absolute

dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:, :].values
y = dataset.iloc[:, -2].values

#misingvalues
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:8])
X[:, 0:8] = imputer.transform(X[:, 0:8])

#onehotencoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [9])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#train and test spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:,:-1], y, test_size = 0.2, random_state = 1)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train= sc.fit_transform(y_train.reshape(-1, 1))
y_test = sc.transform(y_test.reshape(-1, 1))


#--graphs--#

plt.scatter(X_train[:,5],y_train,color='r',alpha=0.01)
plt.scatter(X_train[:,6],y_train,alpha=0.01)
plt.show()
plt.scatter(X_train[:,7],y_train,alpha=0.01)
plt.scatter(X_train[:,8],y_train,alpha=0.01)
plt.show()
plt.scatter(X_train[:,9],y_train,alpha=0.01)
plt.scatter(X_train[:,10],y_train,alpha=0.01)
plt.show()
plt.scatter(X_train[:,11],y_train,alpha=0.01)
plt.scatter(X_train[:,12],y_train,alpha=0.01)
plt.show()

#--Histogramms--#

for i in range(5,13):
   if i == 5: 
       plt.title('Longitude')
   if i == 6: 
       plt.title('Latitude')
   if i == 7: 
       plt.title('Housing_median_age')
   if i == 8: 
       plt.title('Total_rooms')
   if i == 9: 
       plt.title('Total_bedrooms')
   if i == 10: 
       plt.title('Population')
   if i == 11: 
       plt.title('Households')
   if i == 12: 
       plt.title('Median_income_value')
   if i == 13: 
       plt.title('Median_house_value')    
   plt.hist(X[:, i], bins=25, density=False, alpha=0.6, color='b')
   plt.show()
   
plt.hist(dataset.iloc[:,-1], bins=25, density=False, alpha=0.6, color='b')
plt.title('Ocean_proximity')
plt.show()
 


#multiple linear regression

regressor = LinearRegression()
model_multi=regressor.fit(X_train, y_train)
linear_test_prediction = model_multi.predict(X_test)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_pred),1)),1))

import tensorflow as tf


#---ερωτημα 1---#

neur = tf.keras.models.Sequential()
neur.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal', activation='linear'))
neur.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics= ['mse','mae'] )
history = neur.fit(X_train, y_train, batch_size = 32,epochs = 100, validation_data=(X_test, y_test))

plt.plot(history.history['mae'])
plt.plot(history.history['mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['mean absolute error','mean squared error'], loc='upper left')

plt.show()

test_predictions = neur.predict(X_test)
accuracies = cross_val_score(LinearRegression(), y_pred, test_predictions, cv= 10)
acc=mean(absolute(accuracies))

print("mean absolute error Accuracy: {:.2f} %".format(acc*100))
print("least mean error Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

plt.scatter(y_pred, test_predictions,marker=".")
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-2, 2], [-2,2])
plt.show()



#---ερωτημα 3---#


neur = tf.keras.models.Sequential()
neur.add(tf.keras.layers.Dense(units=64, kernel_initializer='normal', activation='relu'))
neur.add(tf.keras.layers.Dense(units=64, kernel_initializer='normal', activation='relu'))
neur.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal', activation='relu'))
neur.compile(optimizer= 'adam', loss= 'mean_squared_error', metrics= ['mse','mae'] )
scores=neur.evaluate(X_test,y_test)
history = neur.fit(X_train, y_train, batch_size = 32,epochs = 100, validation_data=(X_test, y_test))
print("%s: %.2f%%" % (neur.metrics_names[1], scores[1]*100))

test_predictions = neur.predict(X_test)

accuracies = cross_val_score(LinearRegression(), y_pred, test_predictions, cv= 10)

acc=mean(absolute(accuracies))

print("mean absolute error Accuracy: {:.2f} %".format(acc*100))
print("least mean error Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

plt.plot(history.history['mae'])
plt.plot(history.history['mse'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['mean absolute error','mean squared error'], loc='upper left')
plt.show()


plt.scatter(y_pred, test_predictions)
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-2, 2], [-2,2])
plt.show()







