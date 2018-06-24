had nn# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:52:13 2018

@author: akansal2
"""


#for continous variable, we draw first box plot to check the distribution , if it is not normal sitstribution, take log or sqrt
#for discrete variable, it shud be monotonus i,e it shud be able to classify the data independently, if not, make it do thos. or other wise leave the feature
 





#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


np.set_printoptions(threshold = 800)     #to increase the view of ndarray
#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Loan Prediction 3\\train_u6lujuX_CVtuZ9i.csv')

#analysing Data set
Dataset.describe()
print(Dataset.head(20))
print(Dataset.iloc[:,:].isnull().sum())


"""
#handling missing values
#find the mean and mode for all the spcecific columns
Gender_mode = str(Dataset.iloc[:,1].mode())
Married_mode = str(Dataset.iloc[:,2].mode())
Dependents_mode = str(Dataset.iloc[:,3].mode())
SelfEmployed_mode = str(Dataset.iloc[:,5].mode())
LoanAmount_mean = Dataset.iloc[:,8].mean()
LoanAmount_term =  int(Dataset.iloc[:,9].mean())
CreditHistory_mode = int(Dataset.iloc[:,10].mode())

#filling na
Dataset.iloc[:,1] = Dataset.iloc[:,1].fillna(Gender_mode)
Dataset.iloc[:,2] = Dataset.iloc[:,2].fillna(Married_mode)
Dataset.iloc[:,3] = Dataset.iloc[:,3].fillna(Dependents_mode)
Dataset.iloc[:,5] = Dataset.iloc[:,5].fillna(SelfEmployed_mode)
Dataset.iloc[:,8] = Dataset.iloc[:,8].fillna(LoanAmount_mean)
Dataset.iloc[:,9] = Dataset.iloc[:,9].fillna(LoanAmount_term)
Dataset.iloc[:,10] = Dataset.iloc[:,10].fillna(CreditHistory_mode)

"""

#filling na
Dataset.iloc[:,1] = Dataset.iloc[:,1].fillna(str(Dataset.iloc[:,1].mode()[0]))
Dataset.iloc[:,2] = Dataset.iloc[:,2].fillna(str(Dataset.iloc[:,2].mode()[0]))
Dataset.iloc[:,3] = Dataset.iloc[:,3].fillna(str(Dataset.iloc[:,3].mode()[0]))
Dataset.iloc[:,5] = Dataset.iloc[:,5].fillna( str(Dataset.iloc[:,5].mode()[0]))
Dataset.iloc[:,8] = Dataset.iloc[:,8].fillna(Dataset.iloc[:,8].mean())
Dataset.iloc[:,9] = Dataset.iloc[:,9].fillna(Dataset.iloc[:,9].mean())
Dataset.iloc[:,10] = Dataset.iloc[:,10].fillna(int(Dataset.iloc[:,10].mode()[0]))

#checking again if something is nan
print(Dataset.iloc[:,:].isnull().sum())



#Breaking down into X and y
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values




#converting categorical to label encoder
from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
label_encoder_X1 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1])
label_encoder_X2 = LabelEncoder()
X[:,2] = label_encoder_X2.fit_transform(X[:,2])
label_encoder_X3 = LabelEncoder()
X[:,3] = label_encoder_X3.fit_transform(X[:,3])
label_encoder_X4 = LabelEncoder()
X[:,4] = label_encoder_X4.fit_transform(X[:,4])
label_encoder_X5 = LabelEncoder()
X[:,5] = label_encoder_X5.fit_transform(X[:,5])
label_encoder_X11 = LabelEncoder()
X[:,11] = label_encoder_X11.fit_transform(X[:,11])



#removing unncessary featues
X = np.delete(X,0,1)

#applyying onehot encoder
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(categorical_features = [0,1,2,3,4,9,10])
X = OHE.fit_transform(X).toarray()

#dividing training and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)




#applying Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#X = sc_X.fit_transform(X)

#
##fitting classifier
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state= 0)
#classifier.fit(X_train,y_train)


#fitting other classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', degree = 2,random_state = 0)
classifier.fit(X_train,y_train)



##fitting KNN
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5,metric = 'minkowski',p=2)
#classifier.fit(X_train,y_train)


#predicting y
y_pred = classifier.predict(X_test)
#y_pred1 = classifier.predict(X)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



#applying on actual test data
Dataset_test = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Loan Prediction 3\\test_Y3wMUE5_7gLdaTN.csv')

#analysing Data set
Dataset_test.describe()
Dataset_test.head(20)
print(Dataset_test.iloc[:,:].isnull().sum())


#filling na
Dataset_test.iloc[:,1] = Dataset_test.iloc[:,1].fillna(str(Dataset_test.iloc[:,1].mode()[0]))
Dataset_test.iloc[:,2] = Dataset_test.iloc[:,2].fillna(str(Dataset_test.iloc[:,2].mode()[0]))
Dataset_test.iloc[:,3] = Dataset_test.iloc[:,3].fillna(str(Dataset_test.iloc[:,3].mode()[0]))
Dataset_test.iloc[:,5] = Dataset_test.iloc[:,5].fillna( str(Dataset_test.iloc[:,5].mode()[0]))
Dataset_test.iloc[:,8] = Dataset_test.iloc[:,8].fillna(Dataset_test.iloc[:,8].mean())
Dataset_test.iloc[:,9] = Dataset_test.iloc[:,9].fillna(Dataset_test.iloc[:,9].mean())
Dataset_test.iloc[:,10] = Dataset_test.iloc[:,10].fillna(int(Dataset_test.iloc[:,10].mode()[0]))

#checking again if something is nan
print(Dataset_test.iloc[:,:].isnull().sum())

#getting featues
X_actualtest = Dataset_test.iloc[:,:].values

#for new test data (label enconder and onehot encoder)
X_actualtest[:,1] = label_encoder_X1.transform(X_actualtest[:,1])
X_actualtest[:,2] = label_encoder_X2.transform(X_actualtest[:,2])
X_actualtest[:,3] = label_encoder_X3.transform(X_actualtest[:,3])
X_actualtest[:,4] = label_encoder_X4.transform(X_actualtest[:,4])
X_actualtest[:,5] = label_encoder_X5.transform(X_actualtest[:,5])
X_actualtest[:,11] = label_encoder_X11.transform(X_actualtest[:,11])


#removing unncessary featues
X_actualtest = np.delete(X_actualtest,0,1)


#applyying onehot encoder
X_actualtest = OHE.transform(X_actualtest).toarray()


#applying Feature scaling
X_actualtest = sc_X.transform(X_actualtest)


#predicting y
y_pred = label_encoder_y.inverse_transform(classifier.predict(X_actualtest))
A = pd.DataFrame(y_pred)

from openpyxl import load_workbook
writer = pd.ExcelWriter('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Loan Prediction 3\\Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()




