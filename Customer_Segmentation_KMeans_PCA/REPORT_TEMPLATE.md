# Final Project Report
## Customer Segmentation Using K-Means + PCA

**Student Name:** ____________________  
**Batch:** 3  
**Subject:** Artificial Intelligence  
**Project Question:** 3  
**Submission Date:** ____________________

## 1. Introduction
Customer segmentation is the process of dividing customers into groups with similar purchasing behaviour. Businesses can use these groups to plan marketing campaigns, improve customer retention and make better business decisions.

## 2. Problem Statement
The objective is to segment customers into meaningful groups using their purchasing behaviour and an unsupervised machine-learning algorithm.

## 3. Objectives
- Load and prepare customer transaction data.
- Handle missing and infinite values.
- Create customer-level purchasing features.
- Scale the features before clustering.
- Apply K-Means and use the Elbow Method to guide the choice of K.
- Use PCA to visualize clusters in two dimensions.
- Explain the business meaning of each cluster.

## 4. Dataset
**Dataset name:** ____________________  
**Source / URL:** ____________________  
**Number of transaction rows:** ____________________  
**Number of customers after cleaning:** ____________________  
**Important columns:** InvoiceNo, Quantity, InvoiceDate, UnitPrice, CustomerID.

> If using the included sample_transactions.csv, state clearly that it is synthetic demonstration data, not real customer data.

## 5. Data Preprocessing and Feature Engineering
Explain how missing customer IDs, invalid dates, non-positive quantities/prices, cancelled invoices and infinite values were handled. Describe Recency, Frequency, Monetary, TotalQuantity and AverageOrderValue.

## 6. Methodology
### StandardScaler
Explain why scaling is needed before distance-based K-Means.

### Elbow Method
Insert `elbow_plot.png`. State the K values tested and explain where the curve begins to flatten. Record the final K selected: ______.

### K-Means
Explain how K-Means assigns customers to clusters based on feature similarity.

### PCA
Insert `pca_clusters.png`. State the variance explained by the first two principal components: ______.

## 7. Results
Insert `cluster_sizes.png` and summarize the number of customers in each cluster. Attach or summarize `cluster_interpretation.csv`.

## 8. Business Interpretation
For every cluster, report its average Recency, Frequency and Monetary values. Explain what the group may represent and recommend an appropriate business action. Avoid treating cluster IDs as rankings.

## 9. Limitations
Discuss dataset quality, synthetic versus real data, sensitivity to K and outliers, and information lost when PCA is reduced to two dimensions.

## 10. Conclusion
Summarize what the segmentation reveals and how a business could use the results.

## 11. References
List the dataset source and any learning or documentation sources used.
