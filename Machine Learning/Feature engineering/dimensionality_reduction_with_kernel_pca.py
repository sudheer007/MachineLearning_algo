
# coding: utf-8
---
title: "Dimensionality Reduction With Kernel PCA"
author: "sudheer sandu"
date: 2017-12-20T11:53:49-07:00
description: "How to reduce the dimensions of the feature matrix using kernels for machine learning in Python."
type: technical_note
draft: false
---
# <a alt="Dimensionality Reduction With Kernel PCA" href="https://machinelearningflashcards.com">
#     <img src="/images/machine_learning_flashcards/Kernel_PCA_print.png" class="flashcard center-block">
# </a>

# ## Preliminaries

# In[1]:


# Load libraries
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles


# ## Create Linearly Inseparable Data

# In[2]:


# Create linearly inseparable data
X, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)


# ## Conduct Kernel PCA

# In[3]:


# Apply kernal PCA with radius basis function (RBF) kernel
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
X_kpca = kpca.fit_transform(X)


# ## View Results

# In[4]:


print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kpca.shape[1])

