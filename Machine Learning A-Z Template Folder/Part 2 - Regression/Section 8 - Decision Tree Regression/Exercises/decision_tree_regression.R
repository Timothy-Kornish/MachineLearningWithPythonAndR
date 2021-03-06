# Decision Tree Regression

library(e1071)
library(ggplot2)
library(rpart)

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
#                 Fitting Decision Tree Regression to the dataset
#-------------------------------------------------------------------------------

regressor = rpart(formula = Salary ~ .,
                data = dataset,
                control = rpart.control(minsplit = 3))
summary(regressor)

#-------------------------------------------------------------------------------
#             Predicting values for Decision Tree Regression
#-------------------------------------------------------------------------------

y_pred = predict(regressor, data.frame(Level = 6.5))
print("=======================================")
print(y_pred)
print("=======================================")

#-------------------------------------------------------------------------------
#             Visualizing Decision Tree Regression model to the dataset
#-------------------------------------------------------------------------------

png('decision_tree_regression.png')
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
decision_tree_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level=x_grid))), colour = 'blue') +
  ggtitle('Salary prediction (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
plot(decision_tree_plot)
dev.off()
