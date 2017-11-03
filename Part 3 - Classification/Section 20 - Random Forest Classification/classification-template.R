# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

#fitting logistic regression to our trainingset
classifier = 

#predicting test set results
  y_pred = predict(classifier, newdata = test_set)

#building confusion matrix or evaluating
cm = table(test_set[,3],y_pred)

#visualising the training_set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
x1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
x2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
grid_set = expand.grid(x1,x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'logistic regression (training set)',
     xlab = 'Age' ,ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
contour(x1,x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 19, col = ifelse(set[, 3] == 1, 'green4', 'red3'))


#visualising the test_set results
set = test_set
x1 = seq(min(set[,1])-1, max(set[,1])+1, by = 0.01)
x2 = seq(min(set[,2])-1, max(set[,2])+1, by = 0.01)
grid_set = expand.grid(x1,x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'logistic regression (training set)',
     xlab = 'Age' ,ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
contour(x1,x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 19, col = ifelse(set[, 3] == 1, 'green4', 'red3'))