#importing the dataset
dataset = read.csv('Mall_Customers.csv')
x = dataset[4:5]

#using the dendogram to find optimal no of clusters
dendogram = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
plot(dendogram,
     main = paste('dendogram'),
     xlab = 'customers',
     ylab = 'euclidean distance')

#fitting the hierarchical clustering to our dataset
hc = hclust(dist(x, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc , 5)

#visualising the clusters
library(cluster)
clusplot(x,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('cluster of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')