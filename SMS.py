#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as p

url = "SMSSpamCollection.txt"
dataset = p.read_table(url, encoding='latin-1', header = None)


# In[2]:


print(dataset.shape)


# In[3]:


dataset.head(10)


# In[4]:


print(dataset.groupby(0).size())


# Displaying the proportion of spam and ham in the dataset taken from kaggle

# In[5]:


import matplotlib.pyplot as plt

# Data to plot
labels = 'Spam', 'Ham'
sizes = [len(dataset[dataset[0] == 'spam']), len(dataset[dataset[0] == 'ham'])]
colors = ['gold', 'green']
explode = (0.2, 0)

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

plt.axis('equal')
plt.show()


# Make the sms in lower case
# Removal of the punctuations
# Removal of stop-words (a, an, of, until...)

# In[6]:


import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def smsTransformation(message):
    message = message.translate(str.maketrans('', '', string.punctuation))
    word = [words.lower() for words in word_tokenize(message) if words.lower() not in stopwords.words("english")
            and words.lower().isalpha()]
    message = ' '.join(word)
    return message


# In[7]:


for i in range(5571):
    dataset.loc[i, 1] = smsTransformation(dataset.loc[i, 1])


# In[8]:


dataset.head(10)


# the words are referenced by numbers, same for the sms row
# It is having a set of (lineNO, wordNO)
# --the whole logic is not understood, the technique is taken from kaggle--

# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#vectorizer = TfidfVectorizer(encoding = "latin-1")
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(dataset[1])


# In[10]:


#print(features.shape)
#print(vectorizer.get_feature_names())
features


# In[11]:


print(features.toarray())


# In[12]:


from sklearn.preprocessing import LabelEncoder

category = LabelEncoder().fit_transform(dataset[0])


# In[13]:


from sklearn.model_selection import train_test_split

xTrain, xValidation, yTrain, yValidation = train_test_split(features, category, test_size = 0.2, random_state = 42)


# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

gaussianNb = MultinomialNB()
gaussianNb.fit(xTrain, yTrain)


# In[15]:


prediction = gaussianNb.predict(xValidation)
accuracy_score(yValidation, prediction) 


# In[16]:


from sklearn.metrics import classification_report

print(classification_report(yValidation, prediction))


# In[17]:


def smsFilter(message):
    message = smsTransformation(message)
    form = vectorizer.transform([message])
    prediction = gaussianNb.predict(form)
    if (prediction):
        return 'spam'
    else:
        return 'ham'
   
    
smsFilter('Uber Eats: Hanging out with friends? Order in what you crave. Use code UBDR1 to get 50% off 4 orders upto Rs.75, TCA. Order now: t.uber.com/leq')


# In[ ]:


from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
   return '''It's a %s''' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        message = request.form['message']
        answer = smsFilter(message)
    return redirect(url_for('success',name = answer))

if __name__ == '__main__':
   app.run()


# In[ ]:




