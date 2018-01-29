#-------------------------------------------------------------------------------
#                            Logistic Regression
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#                            Libraries
#-------------------------------------------------------------------------------

library(caTools)
library(ggplot2)
library(ElemStatLearn)

#-------------------------------------------------------------------------------
#                            Loading in Data
#-------------------------------------------------------------------------------

dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]
print('-------------------------------------------------------------------')
print(dataset)
print('-------------------------------------------------------------------')
print('----------------========== Data Set =============------------------')
summary(dataset)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                            train test split
#                             And Scaling
#-------------------------------------------------------------------------------

set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

print('--------------========== Training Set =============----------------')
summary(training_set)
print('-------------------------------------------------------------------')
print('----------------========== Test Set =============------------------')
summary(test_set)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                  Fitting Logistic Regression to training set
#-------------------------------------------------------------------------------

classifier = glm(formula = Purchased ~.,
                 family = binomial,
                 data = training_set)

#-------------------------------------------------------------------------------
#                  Predicting test set results
#-------------------------------------------------------------------------------

prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])

print('---------========== Probability of Purchase =============----------')
print(prob_pred)
print('-------------------------------------------------------------------')

y_pred = ifelse(prob_pred > 0.5, 1, 0)

print('-----------========== answer of Purchase =============-------------')
print(y_pred)
print('-------------------------------------------------------------------')

print(length(test_set[, 3]))
print(length(prob_pred))
print(length(y_pred))
#-------------------------------------------------------------------------------
#                  Making a Confusion Matrix
#-------------------------------------------------------------------------------

cm = table(test_set[, 3], y_pred)
print('-----------========== Confusion Matrix =============---------------')
print(cm)
print('-------------------------------------------------------------------')

#-------------------------------------------------------------------------------
#                  Visualizing Data on training set
#-------------------------------------------------------------------------------

set = training_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)

grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

png('training_set_Logistic_regression.png')
plot(set[, -3],
     main = 'Logistic Regression (Training Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
dev.off()

#-------------------------------------------------------------------------------
#                  Visualizing Data on test set
#-------------------------------------------------------------------------------

set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)

grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)

png('test_set_Logistic_regression.png')
plot(set[, -3],
     main = 'Logistic Regression (Test Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
dev.off()
