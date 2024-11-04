# CryptoClustering
chllenge 19




## Overview

The Crypto Clustering project aims to predict if cryptocurrencies are affected by 24-hour or 7-day price changes using unsupervised learning techniques, specifically K-means clustering. Additionally, the project explores the impact of dimensionality reduction using Principal Component Analysis (PCA) on clustering.

## Steps

1. Load and preprocess the data.
2. Scale the data using StandardScaler.
3. Find the best value for k using the elbow method.
4. Cluster cryptocurrencies with K-means using the original scaled data.
5. Perform PCA to reduce the features to three principal components.
6. Find the best value for k using the PCA data.
7. Cluster cryptocurrencies with K-means using the PCA data.
8. Visualize and compare the results using hvPlot.

## Results

The project includes the following visualizations:

1. Elbow curve for the original data.

 ![image alt](elbow_curve.png)

2. Elbow curve for the PCA data.

![image alt](elbow_curve_using_PCA.png)


3. Scatter plot of cryptocurrency clusters based on the original data.clusters_based_on_original_data

![image alt](clusters_based_on_original_data.png)


4. Scatter plot of cryptocurrency clusters based on the PCA data.

![image alt](clusters_based_on_PCA_data.png)



## Conclusion

The project analyzes the impact of using fewer features on clustering the data using K-means. Comparing the clustering results of the original data and the PCA data helps to understand the effect of dimensionality reduction on the clustering process.

## Dependencies

- Python
- pandas
- NumPy
- scikit-learn
- hvPlot

=======


