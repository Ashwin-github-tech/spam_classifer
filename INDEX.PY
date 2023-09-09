import numpy as np
import pandas as pd
import seaborn as sbs

df=pd.read_csv("/content/emails.csv")
df.head()

df=df.iloc[:,:2]

df.info()

df.shape

df.isna().sum()

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

df.isnull().sum()

df.isnull().values.any()

df.shape
x=df['text']
y=df['spam']

x.head()

y.head()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x)

x.toarray()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4,random_state=1)

from sklearn.naive_bayes import MultinomialNB
gnb=MultinomialNB()
gnb.fit(xtrain,ytrain)
gnb.score(xtest,ytest)

from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(xtrain,ytrain)
bnb.score(xtest,ytest)

new_emails = ['your_input']
X_new = cv.transform(new_emails)
y_new = gnb.predict(X_new)

print(y_new)

for i in y_new:
  if i == '1':
    print("SPAM")
  else:
    print("NOT SPAM")
