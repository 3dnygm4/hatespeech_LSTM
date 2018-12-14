
# coding: utf-8

# # Minor Project

# ### Importing the required libraries

# In[3]:


import plaidml.keras
plaidml.keras.install_backend()


# In[4]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
#get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the data using pandas

# In[5]:


df = pd.read_csv("./labeled_data.csv")


# In[6]:


df.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'],axis=1,inplace=True)
df.info()


# ### Columns key:
# count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
# 
# 
# hate_speech = number of CF users who judged the tweet to be hate speech.
# 
# 
# offensive_language = number of CF users who judged the tweet to be offensive.
# 
# 
# neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
# 
# 
# class = class label for majority of CF users.
# 
#     0 - hate speech
#     1 - offensive  language
#     2 - neither
# 
# tweet = raw tweet text

# In[7]:


df = df.rename(index=str, columns={"class": "op", "tweet": "tweet"})
df.head()


# In[8]:


df['op'].hist()


# This histogram shows the imbalanced nature of the task - most tweets containing "hate" words as defined by Hatebase were 
# only considered to be offensive by the CF coders. More tweets were considered to be neither hate speech nor offensive language than were considered hate speech.

# In[9]:


X = df.tweet
Y = df.op
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
Y = to_categorical(Y)
#Y = Y.reshape(-1,1)
print(Y)
print(Y[85])


# In[10]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)


# In[11]:


max_words = 1000
max_len = 260
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


# ### Defining the Model

# In[12]:


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(3,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[13]:


model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['accuracy'])


# ### Custom weight assignment due to class imbalance problem

# In[14]:


class_weight = {0: 50.,
                1: 1.,
                2: 20.}


# ### Training the model

# In[23]:


model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10, class_weight=class_weight)


# In[20]:


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


# ### Evaluating the accuracy of the model

# In[87]:


accr = model.evaluate(test_sequences_matrix,Y_test)


# In[88]:


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[ ]:


y_preds = model.predict(test_sequences_matrix)


# In[90]:


Y_t = []
for i in Y_test:
  Y_t.append(np.argmax(i))
  
#print(Y_t)


Y_p = []

for i in y_preds:
  Y_p.append(np.argmax(i))
  
#print(Y_p)

  


# ### Displaying the classfication summary

# In[91]:


report = classification_report( Y_t, Y_p )
print(report)


# ### Analyzing the confusion matrix 

# In[92]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_t,Y_p)
matrix_proportions = np.zeros((3,3))
for i in range(0,3):
    matrix_proportions[i,:] = confusion_matrix[i,:]/float(confusion_matrix[i,:].sum())
names=['Hate','Offensive','Neither']
confusion_df = pd.DataFrame(matrix_proportions, index=names,columns=names)
plt.figure(figsize=(5,5))
sns.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
plt.ylabel(r'True categories',fontsize=14)
plt.xlabel(r'Predicted categories',fontsize=14)
plt.tick_params(labelsize=12)

