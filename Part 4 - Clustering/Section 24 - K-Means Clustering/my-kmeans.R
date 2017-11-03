#importing the dataset
dataset = read.csv('mall.csv')
x = dataset[4:5]

#using the elbow method
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(x,i)$withinss)
plot(1:10,wcss,type = 'b',main = paste('cluster of clients'),xlab = 'number of clusters',ylab = 'wcss')

#applying k-means to the dataset
kmeans = kmeans(x,5,iter.max = 300,nstart=10)

# Visualising the clusters
library(cluster)
clusplot(x,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')