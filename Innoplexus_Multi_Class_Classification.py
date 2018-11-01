# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:09:20 2018

@author: akansal2
"""

####################################################################
#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import random
from sklearn.utils import shuffle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score





#####################################################################
#importing dataset
df_train = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/Innoplexus challenge/train_data.csv')
df_test = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/Innoplexus challenge/test.csv')

'''
#additional code to read huge html file
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
df_html = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/Innoplexus challenge/html_data.csv',engine='python')



'''


#################################################################################
#data and problem analysis
df_train.shape  #(53447, 4)
#df_html.shape  # (79345, 2)
df_test.shape   #(25787, 3)


df_train.head(5)
'''
df_html['Webpage_id'][0]
df_html['Html'][0]


#checking frequency distribution
df_train['Tag'].value_counts().plot(kind ='bar')


#checking of domain is repeated for different tags
df_train[df_train['Domain'] == 'www.ferring.com']['Tag']
'''



####################################################################################
#modifying data frame and preprocessing
#things to do
#shuffling the data
df_train = shuffle(df_train)


#????????????????????????????
#oversampling and undersampling of categories - Imbalanced data  - We wiill try is later 
#????????????????????????????




#preprocessing of URL and Domain name

#Domain name pre-processing
def get_domain_name_content(domain):
    content = domain.split('.')
    #content.remove(content[-1])
    content = [x for x in content if len(x) >2]
    stopwords = ['www','org','com']
    content = [word for word in content if word not in stopwords]
    content = ' '.join(content)
    return content


df_train['Domain'] = df_train['Domain'].apply(lambda x : get_domain_name_content(x))



#processing URL name and fetching information from it
#identify what feature from URL align it to particular category

  '''
  1) People profile-  profile Name_entity at the end /Aditya-Kansal or /Aditya_kansal,people,dr
  2) Conferences/Congress conference
  3) Forums forums,forum,senetence with - in URLS,blog,community,discussion,tpoic
  4) News article news (use regex to identofy News, news,/news/ news=times and so on),press,release
  5) Clinical trials trial trials R0000 or C0000
  6) Publication - content articles article doi "87/12" pattern   .long,10.1186
  7) Thesis     - Edu ,library,caltech,columbia,handle,full,catalog
  8) Guidelinesclinicalguidelines, guidance guidelines clinical-guidelines,summar,
  9) Others
'''


regexp_edu = 'edu'
regexp_news = 'news'
regexp_forum = 'forum'
regexp_guid = 'guid'
regexp_content = 'content'
regexp_article = 'article'
regexp_doi = 'doi'
regexp_long = '.long'
regexp_trial = '.trial'
regexp_clinic = 'clinic'
regexp_R000 = '[a-z]000'
regexp_profile = 'profile'
regexp_summar = 'summar'
regexp_catalog = 'catalog'
regexp_full = 'full'
regexp_handle = 'handle'
regexp_columbia = 'columbia'
regexp_caltech = 'caltech'
regexp_library = 'librar'
regexp_10886 = '[0-9][0-9]\.[0-9][0-9][0-9][0-9]'
regexp_press = 'press'
regexp_release = 'release'
regexp_topic = 'topic'
regexp_discussion = 'discussion'
regexp_community = 'community'
regexp_blog = 'blog'
regexp_conferen = 'conferen'
regexp_people = 'people'
regexp_dr = 'dr'
regexp_name = '[/][a-z]+[-][a-z]+[a-z]*[-]?[a-z]*[a-z]*[-]?[a-z]*[a-z]*[-]?[a-z]*[a-z]*[-]?[a-z]*[a-z]*[-]?[a-z]*'



def extract_features(url,regexp):
    url = url.lower()
    return bool(re.search(regexp, url))


def create_features(df,name,regexp):
    df[name] = df['Url'].apply(lambda x : 1 if extract_features(x,regexp) else 0)
    return df[name]

name_to_reg = {'edu':regexp_edu,'news':regexp_news,'forum':regexp_forum,'guid':regexp_guid,'content':regexp_content,
               'article':regexp_article,'doi':regexp_doi,'long':regexp_long,'trial':regexp_trial,
               'clinic':regexp_clinic,'R000':regexp_R000,'profile':regexp_profile,
               'summar':regexp_summar,'catalog':regexp_catalog,'full':regexp_full,'handle':regexp_handle,
               'columbia':regexp_columbia,'caltech':regexp_caltech,'librar':regexp_library,'10886':regexp_10886,
               'press':regexp_press,'release':regexp_release,'topic':regexp_topic,'discussion':regexp_discussion,
               'community':regexp_community,'blog':regexp_blog,'conferen':regexp_conferen,'people':regexp_people,
               'dr':regexp_dr,'name':regexp_name
               }


for name in name_to_reg:
    df_train[name] = create_features(df_train,name,name_to_reg[name])


  
#analyse html


#############################################################################################
#creatng X and Y matrix

#label preprocessing
le_y = LabelEncoder()
Y = le_y.fit_transform(df_train['Tag'])

#first try only with domain , then Domain and URL and then try with domain,URL and  html

#only domain
vect = TfidfVectorizer(lowercase = True,min_df=2,max_features = 2500)
X = vect.fit_transform(df_train['Domain'])


#concatenating url and domain
from scipy.sparse import coo_matrix, hstack
A1 = coo_matrix(X)
A2 = coo_matrix(df_train.iloc[:,4:])
X = hstack([A1,A2]).toarray()





###############################################################################################
#test_train_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,stratify = Y)

###############################################################################################
#model fitting


#KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 10,metric = 'euclidean')
classifier1.fit(X,Y)




#Multilnominal NB
from sklearn.naive_bayes import MultinomialNB
classifier2 = MultinomialNB()
#classifier.fit(X_train,Y_train)
classifier2.fit(X,Y)



#Linear SVC
from sklearn.svm import SVC
classifier3 = SVC(kernel ='rbf')
classifier3.fit(X, Y)

'''
#Adaboost
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier()
classifier.fit(X_train,Y_train)
'''

#Decision tree
from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion = 'entropy')
classifier4.fit(X,Y)





############################################################################################
#messauring F1 score for training data
'''
y_pred_train = classifier.predict(X_train)
score = f1_score(Y_train, y_pred_train, average='weighted')
print(score)
'''





##################################################################################################
#predicting output with the model and checkng f1 score
'''
y_pred_test = classifier.predict(X_test)
score = f1_score(Y_test, y_pred_test, average='weighted')
print(score)
'''


###################################################################################################
#predicting score for final outout
df_test['Domain'] = df_test['Domain'].apply(lambda x : get_domain_name_content(x))

for name in name_to_reg:
    df_test[name] = create_features(df_test,name,name_to_reg[name])

#X_actual_test = vect.transform(df_test['Domain']).todense()
X_actual_test = vect.transform(df_test['Domain'])
A1 = coo_matrix(X_actual_test)

A2 = coo_matrix(df_test.iloc[:,3:])
X_actual_test = hstack([A1,A2]).toarray()




y_actual_pred1 = classifier1.predict(X_actual_test)
y_actual_pred2 = classifier2.predict(X_actual_test)
y_actual_pred3 = classifier3.predict(X_actual_test)
y_actual_pred4 = classifier4.predict(X_actual_test)




#calculate y_actualpred by taking mode

y_actual_pred_text = le_y.inverse_transform(y_actual_pred)


##############################################################################################
#writing it to output
A = pd.DataFrame(y_actual_pred_text)
from openpyxl import load_workbook
writer = pd.ExcelWriter('C:/A_stuff/Learning/Machine Learning/Analytics Vidhya/Innoplexus challenge/Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()
















































