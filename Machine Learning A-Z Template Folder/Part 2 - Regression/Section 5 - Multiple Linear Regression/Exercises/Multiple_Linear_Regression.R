#Multiple Linear Regression
library(caTools)


#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = read.csv('50_Startups.csv')
print('-------------------------------------------------------------------')
print(dataset)
print('-------------------------------------------------------------------')
summary(dataset)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                            Encoding Data
#-------------------------------------------------------------------------------

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))
summary(dataset)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                            train test split
#-------------------------------------------------------------------------------

set.seed(123)
split = sample.split(dataset, SplitRatio = 0.8)
training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)

#-------------------------------------------------------------------------------
#     Fitting Multiple Linear Regression Model for all independent variables
#-------------------------------------------------------------------------------

regressor = lm(formula = Profit ~ ., data = training_set )
summary(regressor)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#     Predicting the test set results
#-------------------------------------------------------------------------------

y_prediction = predict(regressor, newdata = test_set)
summary(y_prediction)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
# Building the optimal model using Backward Elimination with multiple Linear Regression
#-------------------------------------------------------------------------------

back_regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                    data = dataset)
summary(back_regressor)
print('-------------------------------------------------------------------')

back_regressor = lm(formula = Profit ~ R.D.Spend +  Marketing.Spend,
                    data = dataset)
summary(back_regressor)
print('-------------------------------------------------------------------')

back_regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(back_regressor)
print('-------------------------------------------------------------------')
