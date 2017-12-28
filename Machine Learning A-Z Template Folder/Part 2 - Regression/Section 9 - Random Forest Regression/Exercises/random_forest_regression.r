# Random Forest Regression

library(e1071)
library(randomForest)
library(ggplot2)

#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
print('-------------------------------------------------------------------')
print(dataset)
print('-------------------------------------------------------------------')
summary(dataset)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                 Fitting Random Forest Regression to the dataset
#-------------------------------------------------------------------------------

set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)
summary(regressor)

#-------------------------------------------------------------------------------
#             Predicting values for Decision Tree Regression
#-------------------------------------------------------------------------------

y_pred = predict(regressor, data.frame(Level = 6.5))
print("=======================================")
print(y_pred)
print("=======================================")

#-------------------------------------------------------------------------------
#             Visualizing Random Forest Regression model to the dataset
#-------------------------------------------------------------------------------

png('random_forest_regression.png')
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
random_forest_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level=x_grid))), colour = 'blue') +
  ggtitle('Salary prediction (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')
plot(random_forest_plot)
dev.off()
