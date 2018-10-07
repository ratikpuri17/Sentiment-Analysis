
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


# In[4]:


# from google.colab import files
# uploaded = files.upload()


# In[5]:


# from google.colab import files
# uploaded = files.upload()


# In[4]:


df=pd.read_csv("0000000000002747_training_twitter_x_y_train.csv")


# In[5]:


dftest=pd.read_csv("0000000000002747_test_twitter_x_test.csv")


# In[6]:


dftest.head()


# In[7]:


df.head()


# In[8]:


len(df)


# In[9]:


len(dftest)


# In[10]:


categories=set(df['airline_sentiment'])


# In[11]:


categories


# In[12]:


y=df[df['airline_sentiment']=="positive"]
x=np.array(y['text'])
x


# In[13]:


#yt=df[df['airline_sentiment']=="positive"]
xt=np.array(dftest['text'])
xt


# In[14]:


documents=[]
for categ in categories:
    
    smalldf=df[df['airline_sentiment'] == categ ]
    tweets=np.array(smalldf['text'])
    
    for tweet in tweets:
        
        documents.append((tweet,categ))
     


# In[15]:


from nltk.corpus import stopwords
import string


# In[16]:


stop_words=stopwords.words("english")
stop_words+=list(string.punctuation)


# In[17]:


from nltk import pos_tag


# In[18]:


from nltk.stem import WordNetLemmatizer


# In[19]:


lemm=WordNetLemmatizer()


# In[20]:


lemm.lemmatize("played","v")


# In[21]:


from nltk.corpus import wordnet
def get_simple_pos(pos):
  
    if(pos.startswith("J")):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    


# In[22]:


def clean_word(word):
    output=[]
    l=word.split()
    for w in l:
        if(w.lower() not in stop_words):
            pos=pos_tag([w])
            lemmatized_word=lemm.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            output.append(lemmatized_word.lower())
    return output
    


# In[23]:


documents=[ (clean_word(doc),categ) for doc,categ in documents]


# In[26]:


documents[0]


# In[ ]:


training_docs=documents[:]


# In[ ]:


# training_docs=documents[:8235]
# testing_docs=documents[8235:]


# In[ ]:


# training_docs=documents[:35]
# testing_docs=documents[35:]


# In[ ]:


allwords=[]
for doc in training_docs:
    allwords+=doc[0]


# In[267]:


training_docs[1]


# In[29]:


import nltk
nltk.download('stopwords')


# In[ ]:


freq=nltk.FreqDist(allwords)
common=freq.most_common(3000)
features=[ i[0] for i in common]


# In[ ]:


features


# In[ ]:


def get_feat_dict(words):
    feats={}
    words_set=set(words)
    for w in features:
        feats[w]= w in words_set
    return feats
        


# In[ ]:


training_data=[ (get_feat_dict(doc),categ) for doc,categ in training_docs]
#testing_data=[ (get_feat_dict(doc),categ) for doc,categ in testing_docs]


# In[33]:


len(xt)


# In[ ]:


pdata=[get_feat_dict(i) for i in xt]


# In[ ]:


from nltk import NaiveBayesClassifier


# In[ ]:


clf=NaiveBayesClassifier.train(training_data)


# In[298]:


#nltk.classify.accuracy(clf,testing_data)


# In[38]:


len(pdata)


# In[ ]:


ypred=clf.classify_many(pdata)


# In[ ]:


np.savetxt("sentiment_analysis4.csv",ypred,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis4.csv')


# In[171]:


len(ypred)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from nltk.classify.scikitlearn import SklearnClassifier


# In[41]:


type(training_data)


# In[ ]:


gnb=MultinomialNB()
classifier_sklearn=SklearnClassifier(gnb)


# In[44]:


classifier_sklearn.train(training_data)


# In[ ]:


ypredmnb=classifier_sklearn.classify_many(pdata)


# In[ ]:


np.savetxt("sentiment_analysis_mnb.csv",ypredmnb,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis_mnb.csv')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf_rfc=RandomForestClassifier()


# In[ ]:


classifier_sklearn_rfc=SklearnClassifier(clf_rfc)


# In[47]:


classifier_sklearn_rfc.train(training_data)


# In[ ]:


ypredrfnltk=classifier_sklearn_rfc.classify_many(pdata)


# In[ ]:


np.savetxt("sentiment_analysis5_rfc_nltk.csv",ypredrfnltk,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis5_rfc_nltk.csv')


# In[180]:


# nltk.classify.accuracy(classifier_sklearn_rfc,testing_data)


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf_svm=SVC(C=1000,kernel="rbf")


# In[ ]:


classifier_sklearn_svm=SklearnClassifier(clf_svm)


# In[389]:


classifier_sklearn_svm.train(training_data)


# In[ ]:


ypredsvmnltk=classifier_sklearn_svm.classify_many(pdata)


# In[ ]:


np.savetxt("sentiment_analysis6_svm_nltk.csv",ypredsvmnltk,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis6_svm_nltk.csv')


# In[185]:


# nltk.classify.accuracy(classifier_sklearn_svm,testing_data)


# In[73]:


documents[0]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cats=[ c for doc,c in documents]


# In[ ]:


text_docs=[" ".join(doc) for doc,c in documents]


# In[ ]:


x_t_docs=["".join(d) for d in xt]


# In[ ]:


x_train=text_docs


# In[ ]:


count_vec=CountVectorizer(max_features=2000,ngram_range=(1,2))
a=count_vec.fit_transform(x_train)


# In[ ]:


x_t_docs=count_vec.transform(x_t_docs)


# In[ ]:


b=np.array(a.todense())


# In[174]:


b.shape


# In[175]:


len(a.todense())


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf=SVC(C=1000,kernel="rbf")


# In[43]:


clf.fit(a,cats)


# In[509]:


clf.score(x_test_feats,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier()


# In[226]:


rfc.fit(a,cats)


# In[209]:


rfc.score(x_test_feats,y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr=LogisticRegression()


# In[547]:


lr.fit(a,cats)


# In[ ]:


ypredLR=lr.predict(x_t_docs)


# In[ ]:


np.savetxt("sentiment_analysis_using_LR_final.csv",ypredLR,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis_using_LR_final.csv')


# In[ ]:


ypredrf=rfc.predict(x_t_docs)


# In[ ]:


ypredsvm=clf.predict(x_t_docs)


# In[289]:


ypredsvm=="positive"


# In[ ]:


np.savetxt("sentiment_analysis_using_rfc_final.csv",ypredrf,fmt="%s",delimiter=",")


# In[ ]:


np.savetxt("sentiment_analysis_using_svm_final.csv",ypredsvm,fmt="%s",delimiter=",")


# In[ ]:


np.savetxt("sentiment_analysis3.csv",ypred,fmt="%s",delimiter=",")


# In[ ]:



files.download('sentiment_analysis3.csv')


# In[ ]:


from google.colab import files
files.download('sentiment_analysis_using_rfc_final.csv')


# In[ ]:


from google.colab import files
files.download('sentiment_analysis_using_svm_final.csv')


# In[56]:


x_train[:5]


# In[55]:


len(cats)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


import keras


# In[ ]:


model = Sequential()
#model.add(Embedding(10980,1))
model.add(Dense(256, input_dim=2000, activation='relu',bias_initializer='zeros',kernel_initializer='random_uniform'))
#model.add(Activation('relu'))
#model.add(Dense(256, activation='relu',kernel_initializer='random_uniform')) 
model.add(Dense(3,kernel_initializer='random_uniform'))


# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[ ]:


cats=np.array(cats)


# In[ ]:


cats=np.reshape(cats,(10980,1))


# In[ ]:


allcats=np.array([[-1,-1,-1]])
for i in cats:
  
    if(i=="negative"):
        l=[[1,0,0]]
        allcats=np.append(allcats,l,axis=0)
    elif(i=="positive"):
        l=[[0,1,0]]
        allcats=np.append(allcats,l,axis=0)
    else:
        l=[[0,0,1]]
        allcats=np.append(allcats,l,axis=0)


# In[183]:


allcats[0]


# In[184]:


allcats.shape


# In[ ]:


cats2=[ c[0] for c in cats]


# In[186]:


cats2[:5]


# In[ ]:


cats=np.array(cats)


# In[191]:


model.fit(b,allcats[1:], epochs = 25)


# In[192]:


x_t_docs[:2]


# In[ ]:


ykeras=model.predict(x_t_docs)


# In[194]:


len(ykeras)


# In[195]:


ykeras[:5]


# In[ ]:


ykeras_encoded=[ k.max() for k in ykeras]


# In[ ]:


l=[]
for k in range(len(ykeras)):
    m=max(ykeras[k])
    idx=list(ykeras[k]).index(m)
    if(idx==0):
        l.append("negative")
      
    elif(idx==1):
        l.append("positive")
    else:
        l.append("neutral")


# In[ ]:


np.savetxt("sentiment_analysis_using_nn.csv",l,fmt="%s",delimiter=",")


# In[ ]:


from google.colab import files
files.download('sentiment_analysis_using_nn.csv')

