# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 09:29:54 2018

@author: akansal2
"""


####################################################################
#importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,VotingClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier


#####################################################################
#importing dataset
train = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/WNS Promotion Problem/train.csv')
test = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/WNS Promotion Problem/test.csv')

########################################################################
#data exploration
train.describe()
train.groupby(['awards_won?','KPIs_met >80%'])['awards_won?'].count()


############################################################################
#creating new feature
'''
#outperformer
train['outperformer'] = train['awards_won?']*train['KPIs_met >80%']
test['outperformer'] = test['awards_won?']*test['KPIs_met >80%']
'''

###########################################################################
#Checking whether train data is balanced or unbalanced
(train.groupby(['is_promoted'])['employee_id'].count()[1]/train.groupby(['is_promoted'])['employee_id'].count()[0])*100

#9.5 perecent are promoted  out of given data set
#highly unbalanced dataset
'''
#Lets  do undersampling of data
positive_label_count = len(train[train['is_promoted']==1])
negative_label_count = len(train[train['is_promoted']==0])


indices_negative_label = train[train['is_promoted']==0].index
indices_postive_label = train[train['is_promoted']==1].index

random_indices = np.random.choice(indices_negative_label,positive_label_count,replace = False )


#undersampled train
undersample_indices = np.concatenate([random_indices,indices_postive_label]) 
train = train.loc[undersample_indices]

#undersampling gave result of 35 score only. Lets try oversampling using SMOTE 
'''

######################################################################################
#Checking null value

train.info()  



#education has null value -- 4.3 percent missing value
int(train['education'].isnull().sum())/int(train.shape[0])*100
int(train['education'].isnull()[train['is_promoted'] ==1].sum())/int(train.shape[0])*100

#Previous year rating has null value - 7.5 percent
int(train['previous_year_rating'].isnull().sum())/int(train.shape[0])*100



test.info()

#education has null value -- 4.4 percent
int(test['education'].isnull().sum())/int(test.shape[0])*100

#Previous year rating has null value - 7.7 percent
int(test['previous_year_rating'].isnull().sum())/int(test.shape[0])*100

'''
#deleting nan
#since nan values are very less in percentage, We are deleting them

def drop_null(df):
    df = df.dropna()
    return df

train = drop_null(train)
test = drop_null(test)

#checking if deleting null values have deleted all promoted ones and made our data more unbalanced
(train.groupby(['is_promoted'])['employee_id'].count()[1]/train.groupby(['is_promoted'])['employee_id'].count()[0])*100

#still 9.5 percent. We are good with deletion
'''

#dropping above idea of deleting nan and now filling them up with mode
train['education'] = train['education'].fillna(str(train['education'].mode()[0]))
train['previous_year_rating'] = train['previous_year_rating'].fillna(str(train['previous_year_rating'].mode()[0]))
test['education'] = test['education'].fillna(str(test['education'].mode()[0]))
test['previous_year_rating'] = test['previous_year_rating'].fillna(str(test['previous_year_rating'].mode()[0]))



#######################################################################################
#treating employee ID
#Since it is of no use, lets delete this column

def drop_id(df):
    df = df.drop(['employee_id'],axis =1)
    return df

train = drop_id(train)
test = drop_id(test)



###################################################################################
#department
train['department'].value_counts().plot(kind = 'bar')
(train['department'].value_counts()/train['department'].value_counts().sum())*100



#checking relation between department and 'is promoted'
train.groupby(['is_promoted','department'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','department'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','department'])['is_promoted'].count()[1])*100

#Looks like few categories have higher chances of getting promoted compared to others
#lets assign cdepartments based on those categories



#common function
def change_category_to_others(df,attribute,minority_list,others):
    df[attribute] = df[attribute].apply(lambda x: others if x in minority_list else x )
    return df


#common funcion for all attributes
def label_encoding_fit_transform(le,df,attribute):
    df[attribute] = le.fit_transform(df[attribute])
    return df

def label_encoding_transform(le,df,attribute):
    df[attribute] = le.transform(df[attribute])
    return df


#common function to create dummies
def create_dummies(df,attribute):
    df_dummies = pd.get_dummies(df[attribute],drop_first = True,prefix = attribute)
    df = pd.concat([df,df_dummies],axis =1)
    df = df.drop([attribute],axis =1)
    return df


'''
#since finance ,HR ,R&D,Legal form low percentage , i would like to group them in one category
minority_list = ['Finance','HR','R&D','Legal']

train = change_category_to_others(train,'department',minority_list,'Other')
test = change_category_to_others(test,'department',minority_list,'Other')
'''

'''
#creating different lists
High_prob_list = ['Analytics','Finance','Operations','Procurement','Technology']
Low_prob_list = ['HR','R&D','Legal','Sales & Marketing']


train = change_category_to_others(train,'department',High_prob_list,'High')
train = change_category_to_others(train,'department',Low_prob_list,'Low')
test = change_category_to_others(test,'department',High_prob_list,'High')
test = change_category_to_others(test,'department',Low_prob_list,'Low')

'''



#changing them to label encoders


#specific for Department

le_depart = LabelEncoder()

train = label_encoding_fit_transform(le_depart,train,'department')
test = label_encoding_transform(le_depart,test,'department') #assuming categorgies in traning and test set are same

'''
#plotting scatter plot
train[['department','is_promoted']].plot(x = train['department'],y = train['is_promoted'],kind= 'scatter')
'''


train = create_dummies(train,'department')
test = create_dummies(test,'department')


'''
train = train.drop(['department'],axis =1)
test = test.drop(['department'],axis =1)
'''


####################################################################################
#region
train['region'].value_counts().plot(kind = 'bar')
(train['region'].value_counts()/train['region'].value_counts().sum())*100



#checking relation between region and 'is promoted'
(train.groupby(['is_promoted','region'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','region'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','region'])['is_promoted'].count()[1])*100).sort_values()

'''
minority_list1 = ['region_31','region_4','region_27','region_16',
                 'region_11','region_28','region_23','region_29']

minority_list2 = ['region_19','region_20','region_32','region_14',
                 'region_17','region_25','region_5','region_10']
                 
minority_list3 = ['region_30','region_6','region_8','region_1',
                 'region_24','region_12','region_21','region_3',
                 'region_9','region_33','region_34','region_18']

train = change_category_to_others(train,'region',minority_list1,'Others1')
test = change_category_to_others(test,'region',minority_list1,'Others1')
train = change_category_to_others(train,'region',minority_list2,'Others2')
test = change_category_to_others(test,'region',minority_list2,'Others2')
train = change_category_to_others(train,'region',minority_list3,'Others3')
test = change_category_to_others(test,'region',minority_list3,'Others3')
'''
'''

#creating region categories based on probability
High_prob_list = ['region_7','region_3','region_22','region_23',
                 'region_28','region_25','region_17','region_4']

Medium_prob_list = ['region_1','region_30','region_13','region_8',
                 'region_2','region_15','region_27','region_10','region_14']

Low_prob_list = ['region_16','region_12','region_26','region_19',
                 'region_20','region_31','region_11','region_6',
                 'region_5','region_21','region_29','region_32',
                 'region_33','region_24','region_18','region_34','region_9']




train = change_category_to_others(train,'region',High_prob_list,'High')
test = change_category_to_others(test,'region',High_prob_list,'High')
train = change_category_to_others(train,'region',Medium_prob_list,'Medium')
test = change_category_to_others(test,'region',Medium_prob_list,'Medium')
train = change_category_to_others(train,'region',Low_prob_list,'Low')
test = change_category_to_others(test,'region',Low_prob_list,'Low')



'''

#label encoding
le_region = LabelEncoder()
train = label_encoding_fit_transform(le_region,train,'region')
test = label_encoding_transform(le_region,test,'region') #assuming categorgies in traning and test set are same


#Dummy variables
train = create_dummies(train,'region')
test = create_dummies(test,'region')

'''
train = train.drop(['region'],axis =1)
test = test.drop(['region'],axis =1)
'''

###########################################################################################################
#education
train['education'].value_counts().plot(kind = 'bar')
(train['education'].value_counts()/train['education'].value_counts().sum())*100

#(train.groupby(['is_promoted','education'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','education'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','education'])['is_promoted'].count()[1])*100).sort_values()

#looks like education is not that great feature to derive the promotion
#lets drop this feature
'''
train = train.drop(['education'],axis =1)
test = test.drop(['education'],axis =1)
'''


#label encoding
le_education = LabelEncoder()
train = label_encoding_fit_transform(le_education,train,'education')
test = label_encoding_transform(le_education,test,'education') #assuming categorgies in traning and test set are same


#Dummy variables
train = create_dummies(train,'education')
test = create_dummies(test,'education')


###########################################################################################################\
#gender
train['gender'].value_counts().plot(kind = 'bar')
(train['gender'].value_counts()/train['gender'].value_counts().sum())*100


(train.groupby(['is_promoted','gender'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','gender'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','gender'])['is_promoted'].count()[1])*100).sort_values()


'''
#relatiive percent
A = (train['gender'].value_counts()/train['gender'].value_counts().sum())
B = (train.groupby(['is_promoted','gender'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','gender'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','gender'])['is_promoted'].count()[1])*100).sort_values()
A*B
'''

#label encoding
le_gender = LabelEncoder()
train = label_encoding_fit_transform(le_gender,train,'gender')
test = label_encoding_transform(le_gender,test,'gender') #assuming categorgies in traning and test set are same

'''
train = train.drop(['gender'],axis =1)
test = test.drop(['gender'],axis =1)
'''




#########################################################################################################
#recruitment channel
train['recruitment_channel'].value_counts().plot(kind = 'bar')
(train['recruitment_channel'].value_counts()/train['recruitment_channel'].value_counts().sum())*100


(train.groupby(['is_promoted','recruitment_channel'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','recruitment_channel'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','recruitment_channel'])['is_promoted'].count()[1])*100).sort_values()




#label encoding
le_rec_channel = LabelEncoder()
train = label_encoding_fit_transform(le_rec_channel,train,'recruitment_channel')
test = label_encoding_transform(le_rec_channel,test,'recruitment_channel') #assuming categorgies in traning and test set are same


#Dummy variables
train = create_dummies(train,'recruitment_channel')
test = create_dummies(test,'recruitment_channel')

'''
train = train.drop(['recruitment_channel'],axis =1)
test = test.drop(['recruitment_channel'],axis =1)
'''

####################################################################################################
#number of trainings
train['no_of_trainings'].hist() #right skewed
train['no_of_trainings'].skew()  #3.4

(train.groupby(['is_promoted','no_of_trainings'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','no_of_trainings'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','no_of_trainings'])['is_promoted'].count()[1])*100).sort_values()


'''
def traning_categories(number):
    if 0 <= number <=3:
        category = 0
    else:
        category = 1
    return category
        
train['no_of_trainings'] = train['no_of_trainings'].apply(lambda x : traning_categories(x))
test['no_of_trainings'] = test['no_of_trainings'].apply(lambda x : traning_categories(x))


train = train.drop(['no_of_trainings'],axis =1)
test = test.drop(['no_of_trainings'],axis =1)        

'''

#################################################################################################
#age
train['age'].hist() # lighright skewed
train['age'].skew()

(train.groupby(['is_promoted','age'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','age'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','age'])['is_promoted'].count()[1])*100).sort_values()

'''
def age_categories(number):
    if 0 <= number <=23:
        category = 1
    elif 24 <= number <=27:
        category = 2
    elif  28<= number <=40:
        category = 3
    elif 41 <= number <=50:
        category = 4
    else:
        category = 5
    return category
        
train['age'] = train['age'].apply(lambda x : age_categories(x))
test['age'] = test['age'].apply(lambda x : age_categories(x))
        

#Dummy variables
train = create_dummies(train,'age')
test = create_dummies(test,'age')
'''
'''

train = train.drop(['age'],axis =1)
test = test.drop(['age'],axis =1)

'''
'''
train['age'] = np.log10(train['age'])
test['age'] = np.log10(test['age'])
'''
######################################################################################################
#previous year rating
train['previous_year_rating'].value_counts().plot(kind = 'bar')
(train['previous_year_rating'].value_counts()/train['previous_year_rating'].value_counts().sum())*100




(train.groupby(['is_promoted','previous_year_rating'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','previous_year_rating'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','previous_year_rating'])['is_promoted'].count()[1])*100).sort_values()



train['previous_year_rating'] = train['previous_year_rating'].apply(lambda x : 120/float(x))
test['previous_year_rating'] = test['previous_year_rating'].apply(lambda x : 120/float(x))

'''
#Dummy variables
train = create_dummies(train,'previous_year_rating')
test = create_dummies(test,'previous_year_rating')
'''


#########################################################################################################
#length of service
train['length_of_service'].hist()
train['length_of_service'].skew()  #1.8


train.groupby(['length_of_service'])['length_of_service'].count()

(train.groupby(['is_promoted','length_of_service'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','length_of_service'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','length_of_service'])['is_promoted'].count()[1])*100).sort_values()

'''

def len_of_service_categories(number):
    if 0 <= number <=5:
        category = 1
    elif 6 <= number <=10:
        category = 2
    elif  10<= number <=15:
        category = 3
    elif 16 <= number <=25:
        category = 4
    else:
        category = 5
    return category
        
train['length_of_service'] = train['length_of_service'].apply(lambda x : len_of_service_categories(x))
test['length_of_service'] = test['length_of_service'].apply(lambda x : len_of_service_categories(x))
  
#Dummy variables
train = create_dummies(train,'length_of_service')
test = create_dummies(test,'length_of_service')

'''

'''
np.log2(train['length_of_service']).skew()  #0.05
train['length_of_service'] = np.log2(train['length_of_service'])
test['length_of_service'] = np.log2(test['length_of_service'])
'''



#########################################################################################################
#KPIS met
train['KPIs_met >80%'].value_counts().plot(kind = 'bar')
(train['KPIs_met >80%'].value_counts()/train['KPIs_met >80%'].value_counts().sum())*100


(train.groupby(['is_promoted','KPIs_met >80%'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','KPIs_met >80%'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','KPIs_met >80%'])['is_promoted'].count()[1])*100).sort_values()


#no tranformation needed


#########################################################################################################
#awards won
train['awards_won?'].hist()
(train['awards_won?'].value_counts()/train['awards_won?'].value_counts().sum())*100


(train.groupby(['is_promoted','awards_won?'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','awards_won?'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','awards_won?'])['is_promoted'].count()[1])*100).sort_values()

#looks important parameters - No change required



########################################################################################################
#avg tranining score


train['avg_training_score'].hist()
(train.groupby(['is_promoted','avg_training_score'])['is_promoted'].count()[1]/(train.groupby(['is_promoted','avg_training_score'])['is_promoted'].count()[0]+ train.groupby(['is_promoted','avg_training_score'])['is_promoted'].count()[1])*100).sort_values()


'''
train['avg_training_score'] = np.log10(train['avg_training_score'])
test['avg_training_score'] = np.log10(test['avg_training_score'])
'''


'''
def score_categories(number):
    if 0 <= number <=30:
        category = 1
    elif 31 <= number <=50:
        category = 2
    elif  51<= number <=60:
        category = 3
    elif 61 <= number <=70:
        category = 4
    elif 71 <= number <=80:
        category = 5
    elif 81 <= number <=85:
        category = 6
    else:
        category = 7
    return category
        
train['avg_training_score'] = train['avg_training_score'].apply(lambda x : score_categories(x))
test['avg_training_score'] = test['avg_training_score'].apply(lambda x : score_categories(x))
 
 
#Dummy variables
train = create_dummies(train,'avg_training_score')
test = create_dummies(test,'avg_training_score')
'''

########################################################################################################
#creating X and Y matrix
Y = train['is_promoted'].iloc[:].reshape(train.shape[0],1)

#droping Y from Traning
train =train.drop(['is_promoted'],axis =1)

#creating X matrix
X = train.values

#######################################################################################################
#oversampling
'''
sm = SMOTE(ratio = 1.0)
X,Y = sm.fit_sample(X,Y)
'''
#######################################################################################################
#checking correlation
#train.groupby(['region','department'])['region'].count()


########################################################################################################
#test train split

#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify = Y)




##########################################################################################################
#model selection
'''
classifier1 = AdaBoostClassifier()
classifier2 = BernoulliNB()
classifier3 = DecisionTreeClassifier()
classifier4 = RandomForestClassifier(n_estimators = 50)
classifier5 = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion = 'entropy',splitter = 'random',max_features = 0.7,),n_estimators = 100 )

classifier = VotingClassifier(estimators = [('boosting',classifier1),('gnb',classifier2),
                                            ('Logistic',classifier3),('Random Forest',classifier4),
                                            ('Bagging',classifier5)],voting = 'hard')


'''    
    
    

'''
classifier1 = DecisionTreeClassifier()
classifier2 = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion = 'entropy',splitter = 'random',max_features = 0.7,),n_estimators = 100 )
classifier3 = RandomForestClassifier(n_estimators = 50)

classifier = VotingClassifier(estimators = [('NB',classifier1),('LR',classifier2),
                                            ('RF',classifier3)],voting = 'hard')

'''   
#classifier = KNeighborsClassifier(n_neighbors = 5,p=1 )    
#classifier = RandomForestClassifier(n_estimators = 100,criterion = 'gini',max_features = 0.9)
classifier = BaggingClassifier(base_estimator =DecisionTreeClassifier(criterion = 'entropy',max_features = 0.6,),n_estimators = 300 )
#classifier = BaggingClassifier(base_estimator =BernoulliNB(),n_estimators = 100 )

#class1 = AdaBoostClassifier()
#classifier = SVC(kernel = 'rbf')
#classifier = BernoulliNB()
#classifier = GaussianNB()
#classifier = ExtraTreesClassifier()
#classifier = SVC(kernel = 'poly')





##########################################################################################################
#fitting and y_predict

#X_train = X
#Y_train = Y





classifier.fit(X_train,Y_train)

#Y_predict_test = classifier.predict(X_test)

Y_predict_train = classifier.predict(X_train)

#######################################################################################################
#evaluating F1 score
train_score = f1_score(Y_train,Y_predict_train,average = 'binary')
print('train score',train_score)

'''
test_score = f1_score(Y_test, Y_predict_test,average = 'binary')
print('test score is ',test_score)
'''


######################################################################################################
#confusion matrix
cm_train = confusion_matrix(Y_train,Y_predict_train)
print(cm_train)
'''
cm_test = confusion_matrix(Y_test,Y_predict_test)
print(cm_test)
'''





######################################################################################################
#final prediction
Y_predict_actual = classifier.predict(test)



########################################################################################################
#writing it to actual output file
A = pd.DataFrame(Y_predict_actual)
from openpyxl import load_workbook
writer = pd.ExcelWriter('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/WNS Promotion Problem/Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()



























































































