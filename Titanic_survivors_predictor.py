# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')
X = dataset.iloc[:,[2,4,5]].values
y = dataset.iloc[:,[1]].values
X_ans = dataset2.iloc[:,[1,3,4]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X_ans = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X_ans[:,1] = labelencoder_X.fit_transform(X_ans[:,1])
#R = X


# Taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])

imputer2 = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2=imputer2.fit(X_ans[:,2:3])
X_ans[:,2:3] = imputer2.transform(X_ans[:,2:3])

# Encoding catagorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,[1,2]] = labelencoder_X.fit_transform(X[:,[1,2]])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X_ans = onehotencoder.fit_transform(X_ans).toarray()

# Splitting data into traing set and test set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

# Training the model
from sklearn.svm import SVC
classifier = SVC(kernel='linear' ,random_state=0 , C=20)
classifier.fit(X_train,y_train)




# Predicting the result
y_pred = classifier.predict(X_test)
y_pred_ans = classifier.predict(X_ans)

# Creating confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_pred,y_test)
accuracy = accuracy_score(y_pred,y_test)




# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, classifier.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, classifier.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()















