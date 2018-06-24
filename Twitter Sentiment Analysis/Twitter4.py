# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:30:56 2018

@author: akansal2
"""

#Importing needed Libraies
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob,Word
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer





#reading csv file and importing it into Datasets
df_train = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Twitter Setiment Analysis\\train_E6oV3lV.csv')
df_test = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Twitter Setiment Analysis\\test_tweets_anuFYb8.csv')




#deleting the unnecessary variables
#here only id is not needed for either training or testing the dataset
df_train = df_train.drop(['id'],axis = 1)
df_test = df_test.drop(['id'],axis = 1)




##########################################################

#function for getting corpus from any Dataframe
def get_corpus(series):
    temp = []
    for i in series:
        temp.append(i)
    return temp    

#getting corpus
corpus_train = get_corpus(df_train.iloc[:,1])
corpus_test = get_corpus(df_test.iloc[:,0])

##################################################
#capturing new features

#smile emoticon
def get_smile_emoticon(corpus):
    smile_emoticon = []
    for i in corpus:
        if 'ð' in i:
            smile_emoticon.append(0)
        else:
            smile_emoticon.append(1)        
    return smile_emoticon

smile_train = get_smile_emoticon(corpus_train)
smile_test = get_smile_emoticon(corpus_test)


################################################################


#getting all ! letters
def get_exclaim_emoticon(corpus):
    exclaim_emoticon = []
    for i in corpus:
        if '!' in i:
            exclaim_emoticon.append(1)
        else:
            exclaim_emoticon.append(0)        
    
    return exclaim_emoticon

exclaim_train = get_exclaim_emoticon(corpus_train)
exclaim_test = get_exclaim_emoticon(corpus_test)



##############################################################

#getting all â¦ letters
def get_angry_emoticon(corpus):    
    angry_emoticon = []
    for i in corpus:
        if 'â¦' in i:
            angry_emoticon.append(1)
        else:
            angry_emoticon.append(0)        

    return angry_emoticon

angry_train = get_angry_emoticon(corpus_train)
angry_test = get_angry_emoticon(corpus_test)



######################################################################

#get_negative words presence
negative_wordlist = ["can't",'no','not','never',"don't","won't","doesn't","hasn't","haven't","hadn't","didn't","shouldn't"]
def get_negative_wordpresence(corpus):    
    negative_word = []
    for i in corpus:
        if i in negative_wordlist:
            negative_word.append(1)
        else:
            negative_word.append(0)  

    return negative_word

negative_train = get_negative_wordpresence(corpus_train)
negative_test = get_negative_wordpresence(corpus_test)


######################################################################33


          
#each word operation
def clean_sent(sentence):
    cleaned_sent = []
    for word in sentence:
        if len(word) > 3 :
            #word = Word(word).spellcheck()[0][0]
            ps = PorterStemmer()
            word = str(ps.stem(word))
            cleaned_sent.append(word)
    return ' '.join(cleaned_sent)

    
#get clean corpus
def clean_corpus(corpus):
    cleaned_corpus = []
    for sent in corpus:
        sent = sent.replace('@user','')
        sent = re.sub('[^a-zA-Z]',' ',sent)
        sent = sent.lower()
        sent = sent.split()
        sent = clean_sent(sent)        
        cleaned_corpus.append(sent)
                
    return cleaned_corpus



corpus_train = clean_corpus(corpus_train)
corpus_test = clean_corpus(corpus_test)


###############################################################

#getting polarity and subjectivty as feature

#polarity and subjectivity
def get_polarity_subj(corpus):
    polarity = []
    subjectivity = []
    for sent in corpus:
        blob = TextBlob(sent)
        
        if blob.sentiment[0] > 0 :
            polarity.append(0)
        else:
            polarity.append(1)
            
        if blob.sentiment[1] > 0 :
            subjectivity.append(0)
        else:
            subjectivity.append(1)

    return polarity,subjectivity
    

pol_train,subj_train = get_polarity_subj(corpus_train)
pol_test,subj_test = get_polarity_subj(corpus_test)


###############################################################

#instead of using count vectorizer , we will use TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer( max_df = 0.2,ngram_range=(1,1),stop_words = 'english')
X = cv.fit_transform(corpus_train).toarray()
y = df_train.iloc[:,0]  

X_actualtest = cv.transform(corpus_test).toarray()



##################################################################
#concatenating arrays

def concat(X,smile,exclaim,angry,negative,pol,sub):
    l = X.shape[0]
    smile = np.array(smile).reshape(l,1)
    exclaim = np.array(exclaim).reshape(l,1)
    angry = np.array(angry).reshape(l,1)    
    negative = np.array(negative).reshape(l,1)
    pol = np.array(pol).reshape(l,1)    
    sub = np.array(sub).reshape(l,1)
    X = np.concatenate((X,smile,exclaim,angry,negative,pol,sub),axis = 1)
    return X

X = concat(X,smile_train,exclaim_train,angry_train,negative_train,pol_train,subj_train)
X_actualtest = concat(X_actualtest,smile_test,exclaim_test,angry_test,negative_test,pol_test,subj_test)


###################################################################

#Now Using train data set , lets put some ML algo
#dividing into test and train data set
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
#


#########################################################################3


##applying random classifier
from sklearn.ensemble import RandomForestClassifier
classifer = RandomForestClassifier(n_estimators = 200,max_features = 0.40,min_samples_leaf = 2,criterion = 'entropy',random_state = 0)
#classifer.fit(X_train,y_train)
classifer.fit(X,y)
#
##applying SVC
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf',random_state=0)
#classifier.fit(X_train,y_train)

#
##applying Bernoulli NB
#from sklearn.naive_bayes import BernoulliNB
#classifer = BernoulliNB()
#classifer.fit(X,y)


'''
#applying decision tree
from sklearn.tree import DecisionTreeClassifier
classifer = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifer.fit(X_train,y_train)
'''
'''
#applying NB classifier
from sklearn.naive_bayes import MultinomialNB
classifer = MultinomialNB()
classifer.fit(X_train,y_train)
'''
#from sklearn.svm import LinearSVC

#predicting y
y_pred = classifer.predict(X_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#calculating F1 score
TP = int(cm[1,1])
TN = int(cm[0,0])
FP = int(cm[0,1])
FN = int(cm[1,0])
recall = TP /(TP+ FN)
precision = TP /(TP+FP)
F1_score = 2* recall*precision/(recall + precision)
print(F1_score)


#predciting for actual X_test
y_pred_actual = classifer.predict(X_actualtest)


#exporting to csv
#A = pd.DataFrame(y_pred_actual)
A = pd.DataFrame(y_pred_actual)
from openpyxl import load_workbook
writer = pd.ExcelWriter('C:\\A_stuff\\Learning\\Machine Learning\\Analytics Vidhya\\Twitter Setiment Analysis\\Output.xls')
A.to_excel(writer,'Sheet2')
writer.save()


















#code for F1 Score




























































