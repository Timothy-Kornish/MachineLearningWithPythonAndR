#Polynomial Regression
library(caTools)
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
#                 Fitting Linear Regression to the dataset
#-------------------------------------------------------------------------------

lin_reg = lm(formula = Salary ~., data = dataset)
summary(lin_reg)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                 Fitting Polynomial Regression to the dataset
#-------------------------------------------------------------------------------

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)
summary(poly_reg)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#             Visualizing Linear Regression model to the dataset
#-------------------------------------------------------------------------------

png('linear_regression.png')
lin_reg_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Salary prediction (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')
plot(lin_reg_plot)
dev.off()

#-------------------------------------------------------------------------------
#             Visualizing Polynomial Regression model to the dataset
#-------------------------------------------------------------------------------

png('polynomial_regression.png')
poly_reg_plot = ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Salary prediction (Polynomial Regression degree = 4)') +
  xlab('Level') +
  ylab('Salary')
plot(poly_reg_plot)
dev.off()

#-------------------------------------------------------------------------------
#             Predicting values for Linear Regression
#-------------------------------------------------------------------------------

lin_predict = predict(lin_reg, data.frame(Level = 6.5))
print('-------------------------------------------------------------------')
summary(lin_predict)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#             Predicting values for Polynomial Regression
#-------------------------------------------------------------------------------

poly_predict = predict(poly_reg, data.frame(Level = 6.5,
                                            Level2 = 6.5^2,
                                            Level3 = 6.5^3,
                                            Level4 = 6.5^4))
summary(poly_predict)
print('-------------------------------------------------------------------')
