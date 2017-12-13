# Simple Linear Regression
library(caTools)
library(ggplot2)


#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = read.csv('Salary_Data.csv')
print('-------------------------------------------------------------------')
print(dataset)
print('-------------------------------------------------------------------')
summary(dataset)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                            train test split
#-------------------------------------------------------------------------------

set.seed(123)
split = sample.split(dataset, SplitRatio = 2/3)
training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)

#-------------------------------------------------------------------------------
#                   Fitting a Linear Regression Model
#-------------------------------------------------------------------------------

regressor = lm(formula = Salary~YearsExperience, data = training_set)
summary(regressor)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                   predicting a Linear Regression Model
#-------------------------------------------------------------------------------

y_pred = predict(regressor, newdata = test_set)
summary(y_pred)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                   Plotting the Linear Regression Model with ggplot
#-------------------------------------------------------------------------------

png('training_set_linear_regression.png')
train_plot = ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')
plot(train_plot)
dev.off()

png('test_set_linear_regression.png')
test_plot = ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')
plot(test_plot)
dev.off()
