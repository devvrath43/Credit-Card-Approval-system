# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 00:35:20 2020

@author: AVICII
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('crx.csv', header = None)
X = dataset.iloc[:, [1,2,7,10,13,14]].values
X1 = pd.DataFrame(X,columns=['A','B','C','D','E','F'])
y = dataset.iloc[:, 15:].values
y1 = pd.DataFrame(y,columns=['Y/N'])
X1 = dataset.iloc[:, [1,2,7,10,13,14]].reset_index(drop = True)

#Handling Missing values
## chnageing all '?' to NaN
array1 = np.array(X)
X = np.where(array1=='?', 'NaN', array1) 
## Changing all values of X to float type
X = np.array(X).astype('float32')

# Filling missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X)
X2 = imputer.transform(X)

X_view = pd.DataFrame(X,columns=['A','B','C','D','E','F'])

''' Removing Outliners '''
Q1 = np.quantile(X2, 0) # dont want to lose first 10%
Q3 = np.quantile(X2, 0.75)
IQR = Q3 - Q1

print(X2 < (Q1 - 1.5 * IQR)) |(X2 > (Q3 + 1.5 * IQR))

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y[: , 0] = labelencoder_y.fit_transform(y[: , 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
y = y[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled= sc.fit_transform(X2)


'''################## Visualization #######################################'''
'''X3 = pd.DataFrame(X2,columns=['A','B','C','D','E','F'])
y2 = pd.DataFrame(y,columns= ['Acceptance'])
data = pd.concat([X3.iloc[: , 0:6], y2.iloc[: , 0:]], axis = 1) '''

X3 = pd.DataFrame(X_scaled,columns=['A','B','C','D','E','F'])
y2 = pd.DataFrame(y,columns= ['Acceptance'])
data = pd.concat([X3.iloc[: , 0:6], y2.iloc[: , 0:]], axis = 1) 


X2 = np.percentile()

# histogram
plt.hist(data)
#plt.legend(loc="upper right")
plt.show()
plt.close()

    
#Scatter plot
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['A'], range(690))
#ax.set_xlabel('Parameters')
#ax.set_ylabel('Acceptance')
#plt.show()

#fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['B'], range(690))
#ax.set_xlabel('Parameters')
#ax.set_ylabel('Acceptance')
#plt.show()

#fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['C'], range(690))
#ax.set_xlabel('Parameters')
#ax.set_ylabel('Acceptance')
#plt.show()

#fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['D'], range(690))
#ax.set_xlabel('Parameters')
#ax.set_ylabel('Acceptance')
#plt.show()

#fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['E'], range(690))
#ax.set_xlabel('Parameters')
#ax.set_ylabel('Acceptance')
#plt.show()

#fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(X3['F'], range(690))
ax.set_xlabel('Parameters')
ax.set_ylabel('Acceptance')
plt.show()


#scatter plot matrix
pd.plotting.scatter_matrix(data)
plt.show()

#Box Plot
data.plot(kind='box', layout= (2,2), sharex=False, sharey=False)
plt.show()

data.plot(kind= 'box', subplots= True, layout= (3,3), sharex=False, sharey=False)
dataset.iloc[:,0:].plot(kind= 'box', subplots= True, layout= (4,4), sharex=False, sharey=False)
plt.show()

#Heatmap
plt.figure(figsize=(9,5))
sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm')


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 0)

'''Classification'''

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
Score_knn = classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)



####################################################################################

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)
Score_svm = classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred)

####################################################################################

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train, y_train)
Score_nb = classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred)

####################################################################################

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
Score_dt = classifier.score(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred)

###################################################################################

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 8, criterion = 'entropy')
classifier.fit(X_train, y_train)
Score_rf = classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred)
#score = classifier.score(X_test, y_pred)




