# SVR

library(e1071)
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
#                 Fitting Support Vector Regression to the dataset
#-------------------------------------------------------------------------------

regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression',
                kernal = 'radial')
summary(regressor)

#-------------------------------------------------------------------------------
#             Predicting values for Support Vector Regression
#-------------------------------------------------------------------------------

y_pred = predict(regressor, data.frame(Level = 6.5))
summary(y_pred)

#-------------------------------------------------------------------------------
#             Visualizing Support Vector Regression model to the dataset
#-------------------------------------------------------------------------------

png('support_vector_regression.png')
lin_reg_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Salary prediction (Support Vector Regression)') +
  xlab('Level') +
  ylab('Salary')
plot(lin_reg_plot)
dev.off()
