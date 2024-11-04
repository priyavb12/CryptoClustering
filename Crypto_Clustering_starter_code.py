#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data

# In[7]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
scaled_data


# In[8]:


# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns = df_market_data.columns
)
df_market_data_scaled
# Copy the crypto names from the original DataFrame
df_market_data_scaled['coin_id'] = df_market_data.index

# Set the coin_id column as index
df_market_data_scaled = df_market_data_scaled.set_index('coin_id')

# Display the scaled DataFrame
df_market_data_scaled.head()



# ---

# ### Find the Best Value for k Using the Original Scaled DataFrame.

# In[11]:


# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))
k


# In[12]:


# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
for i in k:
# 1. Create a KMeans model using the loop counter for the n_clusters
    model = KMeans(n_clusters=i, random_state=0)
# 2. Fit the model to the data using `df_market_data_scaled`
    model.fit(df_market_data_scaled)
# 3. Append the model.inertia_ to the inertia list
    inertia.append(model.inertia_)

inertia


# In[13]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    'k': k,
    'inertia': inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)
df_elbow


# In[14]:


# Plot a line chart with all the inertia values computed with
# the different values of k to visually identify the optimal value for k.
elbow_plot = df_elbow.hvplot.line(x='k', y='inertia', title='Elbow Curve', xticks=k)
elbow_plot


# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** best value for k is 4
# 

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Scaled DataFrame

# In[18]:


# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)


# In[19]:


# Fit the K-Means model using the scaled DataFrame
model.fit(df_market_data_scaled)


# In[20]:


# Predict the clusters to group the cryptocurrencies using the scaled DataFrame
crypto_clusters = model.predict(df_market_data_scaled)
crypto_clusters

# Print the resulting array of cluster values.


# In[21]:


# Create a copy of the scaled DataFrame
df_market_data_scaled_predictions = df_market_data_scaled.copy()


# In[22]:


# Add a new column to the copy of the scaled DataFrame with the predicted clusters
df_market_data_scaled_predictions['crypto_clusters'] = crypto_clusters

# Display the copy of the scaled DataFrame
df_market_data_scaled_predictions


# In[23]:


# Create a scatter plot using hvPlot by setting
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`.
# Color the graph points with the labels found using K-Means and
# add the crypto name in the `hover_cols` parameter to identify
# the cryptocurrency represented by each data point.
clusters_plot = df_market_data_scaled_predictions.hvplot.scatter(
    x = 'price_change_percentage_24h',
    y = 'price_change_percentage_7d' ,
    by = 'crypto_clusters' ,
    hover_cols = ['coin_id'],
    marker = ['hex', 'square', 'cross', 'inverted_triangle'],
    title = 'Crypto Clusters'
)
clusters_plot


# ---

# ### Optimize Clusters with Principal Component Analysis.

# In[26]:


# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# In[27]:


# Use the PCA model with `fit_transform` to reduce the original scaled DataFrame
# down to three principal components.
market_pca_data = pca.fit_transform(df_market_data_scaled)

# View the scaled PCA data
market_pca_data


# In[28]:


# Retrieve the explained variance to determine how much information
# can be attributed to each principal component.
pca.explained_variance_ratio_


# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** 

# In[30]:


# Create a new DataFrame with the PCA data.
df_market_data_pca = pd.DataFrame(
    market_pca_data,
    columns = ['PC1' ,'PC2', 'PC3'])

# Copy the crypto names from the original scaled DataFrame
df_market_data_pca['coin_id'] = df_market_data.index

# Set the coin_id column as index
df_market_data_pca = df_market_data_pca.set_index('coin_id')

# Display the scaled PCA DataFrame
df_market_data_pca


# ---

# ### Find the Best Value for k Using the Scaled PCA DataFrame

# In[33]:


# Create a list with the number of k-values from 1 to 11
k = list(range(1,11))


# In[34]:


# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_pca)
    inertia.append(model.inertia_)


# In[35]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    'k': k,
    'inertia': inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)
df_elbow_pca


# In[36]:


# Plot a line chart with all the inertia values computed with
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot(
    x='k',
    y='inertia',
    title='Elbow Curve using PCA',
    xticks=k
    )
elbow_plot_pca


# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**The best k = 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** No, both values are identical

# ### Cluster Cryptocurrencies with K-means Using the Scaled PCA DataFrame

# In[39]:


# Initialize the K-Means model using the best value for k
model_pca = KMeans(n_clusters=4)


# In[40]:


# Fit the K-Means model using the PCA data
model_pca.fit(df_market_data_pca)


# In[41]:


# Predict the clusters to group the cryptocurrencies using the scaled PCA DataFrame
crypto_cluster_pca = model_pca.predict(df_market_data_pca)

# Print the resulting array of cluster values.
crypto_cluster_pca


# In[42]:


# Create a copy of the scaled PCA DataFrame
df_market_data_pca_predictions = df_market_data_pca.copy()
# Add a new column to the copy of the PCA DataFrame with the predicted clusters
df_market_data_pca_predictions['crypto_cluster'] = crypto_cluster_pca

# Display the copy of the scaled PCA DataFrame
df_market_data_pca_predictions


# In[43]:


# Create a scatter plot using hvPlot by setting
# `x="PC1"` and `y="PC2"`.
# Color the graph points with the labels found using K-Means and
# add the crypto name in the `hover_cols` parameter to identify
# the cryptocurrency represented by each data point.
clusters_plot_pca = df_market_data_pca_predictions.hvplot.scatter(
    x ='PC1',
    y = 'PC2',
    by = 'crypto_cluster',
    hover_cols = ['coin_id'],
    marker = ['hex' , 'square', 'cross', 'inverted_triangle'],
    title = 'crypto'
)
clusters_plot_pca


# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# In[45]:


# Composite plot to contrast the Elbow curves
# YOUR CODE HERE!
elbow_plot + elbow_plot_pca


# In[46]:


# Composite plot to contrast the clusters
# YOUR CODE HERE!
clusters_plot + clusters_plot_pca


# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:We conclude use less featues and get similar performance tothe original model since we can clearly identify four clusters . Additionally reduced number of columns make it easier visualize the value of the clustering tradeoffs is 10 of important information from the data is lossed when doing pca
#  

# In[ ]:




