
# coding: utf-8
---
title: "Using Linear Discriminant Analysis For Dimensionality Reduction"
author: "Sudheer sandu"
date: 2017-12-20T11:53:49-07:00
description: "How to use linear discriminant analysis for dimensionality reduction using Python."
type: technical_note
draft: false
---
# ## Preliminaries

# In[1]:


# Load libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ## Load Iris Data

# In[2]:


# Load the Iris flower dataset:
iris = datasets.load_iris()
X = iris.data
y = iris.target


# ## Create A Linear 

# In[3]:


# Create an LDA that will reduce the data down to 1 feature
lda = LinearDiscriminantAnalysis(n_components=1)

# run an LDA and use it to transform the features
X_lda = lda.fit(X, y).transform(X)


# ## View Results

# In[4]:


# Print the number of features
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])


# ## View Percentage Of Variance Retained By New Features

# In[5]:


## View the ratio of explained variance
lda.explained_variance_ratio_

