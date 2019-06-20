# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:33:18 2019

@author: anish.gautam
"""
#TO PREDICT IF THE REVIEWS FOR RESTURANTS ARE POSITIVE OR NEGATIE
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

data=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t')
formated=[]
for i in range(0,1000):
    rev=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    rev=rev.lower()
    ps=PorterStemmer()
    rev=rev.split()
    rev=[ps.stem(i) for i in rev if i not in stopwords.words('english')]
    rev=' '.join(rev)
    formated.append(rev)
        
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(formated).toarray()
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(Xtrain,ytrain)

ypred=classifier.predict(Xtest)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(ypred,ytest)
acc=accuracy_score(ypred,ytest)


        





    
