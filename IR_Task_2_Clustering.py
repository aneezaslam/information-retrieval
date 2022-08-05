#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[77]:


data = pd.read_csv('/Users/aneezaslam/Desktop/Assessments/News dump.csv')
data.head()


# In[45]:


type(data)


# In[46]:


data.columns


# In[78]:


#creating the tfidf vector
tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(data.Text)
text = tfidf.transform(data.Text)


# In[79]:


print(text)


# In[30]:


def find_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters = k, init_size = 1024, batch_size = 2048, random_state = 20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker = 'o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE vs Cluster Center Plot')
    
find_clusters(text, 20)


# In[37]:



mbkm = MiniBatchKMeans(n_clusters = 3)
mbkmcl = mbkm.fit_predict(text)
data["ClustersMBKM"] = mbkmcl


# In[80]:


#creating the kmeans model
clusters = MiniBatchKMeans(n_clusters = 3, init_size = 1024, batch_size = 2048, random_state = 20).fit_predict(text)


# In[81]:


print (clusters)


# In[29]:


def plt_tsne_pca(data, labels):
    maxlabel = max(labels)
    maxitems = np.random.choice(range(data.shape[0]), size = 3000, replace = True)
    
    pca = PCA(n_components = 2).fit_transform(data[maxitems,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components = 50).fit_transform(data[maxitems,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size = 300, replace = False)
    label_subset = labels[maxitems]
    label_subset = [cm.hsv(i/maxlabel) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize = (14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c = label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c = label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plt_tsne_pca(text, clusters)


# In[27]:


def get_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
            
get_keywords(text, clusters, tfidf.get_feature_names(), 10)


# In[55]:


X = tfidf.transform(["government tax mr blair"])


# In[56]:


prediction = mbkm.predict(X)
print (prediction)

