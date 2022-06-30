#!/usr/bin/env python
# coding: utf-8

# # Predicting Customer Class Using K-Nearest Neighbors

# In[ ]:


get_ipython().system('pip install scikit-learn==0.23.1')


# Let's load required libraries
# 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# <div id="about_dataset">
#     <h2>About the dataset</h2>
# </div>
# 

# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case.
# 
# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
# 
# The target field, called **custcat**, has four possible values that correspond to the four customer groups, as follows:
# 1- Basic Service
# 2- E-Service
# 3- Plus Service
# 4- Total Service
# 
# Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.
# 

# In[14]:


import urllib.request
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
filename = 'teleCust1000t.csv'
urllib.request.urlretrieve(url, filename)


# ### Load Data From CSV File
# 

# In[15]:


df = pd.read_csv('teleCust1000t.csv')
df.head()


# <div id="visualization_analysis">
#     <h2>Data Visualization and Analysis</h2> 
# </div>
# 

# #### Let’s see how many of each class is in our data set
# 

# In[16]:


df['custcat'].value_counts()


# #### 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers
# 

# In[17]:


df.hist(column='income', bins=50)


# ### Feature set
# 

# Let's define feature sets, X:
# 

# In[18]:


df.columns


# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
# 

# In[19]:


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# What are our labels?
# 

# In[20]:


y = df['custcat'].values
y[0:5]


# ## Normalize Data
# 

# In[21]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# ### Train Test Split
# 

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <div id="classification">
#     <h2>Classification</h2>
# </div>
# 

# <h3>K nearest neighbor (KNN)</h3>
# 

# #### Import library
# 

# Classifier implementing the k-nearest neighbors vote.
# 

# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# ### Training
# 
# Let's start the algorithm with k=4 for now:
# 

# In[24]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# ### Predicting
# 
# We can use the model to make predictions on the test set:
# 

# In[28]:


yhat = neigh.predict(X_test)
yhat[0:5]


# ### Accuracy evaluation
# 
# In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
# 

# In[26]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ## Rebuilding the model but with k=6
# 

# In[30]:


# write your code here

k = 6
#Train Model and Predict  
neigh_2 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh_2

yhat = neigh_2.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh_2.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# <details><summary>Click here for the solution</summary>
# 
# ```python
# k = 6
# neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# yhat6 = neigh6.predict(X_test)
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))
# 
# ```
# 
# </details>
# 

# #### What about other K?
# 

# In[31]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# #### Plot the model accuracy for a different number of neighbors.
# 

# In[32]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[33]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

