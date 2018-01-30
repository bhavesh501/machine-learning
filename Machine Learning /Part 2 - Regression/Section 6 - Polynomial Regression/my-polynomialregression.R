#import the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#splitting the dataset into training set and test set
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$dependentvariable,SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

#feature scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#fitting the linear_regression to our dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)

#fitting the polynomial_regression to our dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~., data = dataset)

#visualising the linaer regression
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),color = 'red')+
  geom_line(aes(x=dataset$Level, y=predict(lin_reg,newdata = dataset)),color='blue')+
  ggtitle('salary vs level (linear model)')+
  xlab('level')+
  ylab('salary')
  

#visualising the polynomial regression
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),color = 'red')+
  geom_line(aes(x=dataset$Level, y=predict(poly_reg,newdata = dataset)),color='blue')+
  ggtitle('salary vs level (polynomial model)')+
  xlab('level')+
  ylab('salary')

#predicting with linear regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

#predicting with polynomial regression
y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))