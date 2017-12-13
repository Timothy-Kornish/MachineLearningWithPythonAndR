dataset = read.csv('Data.csv')
print(dataset)

#-------------------------------------------------------------------------------
# filling in missing data
#-------------------------------------------------------------------------------

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                    ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                    dataset$Salary)
print(dataset)

#-------------------------------------------------------------------------------
# encoding categorical data
#-------------------------------------------------------------------------------

dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                        levels = c('No', 'Yes'),
                        labels = c(0, 1))
print(dataset)

#-------------------------------------------------------------------------------
# bringing in library for train test split
#-------------------------------------------------------------------------------

library(caTools)
set.seed(123)

split = sample.split(dataset$Purchased, SplitRatio = 0.8) # SplitRatio is for train_size
print(split)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
print(training_set)
print(test_set)

#-------------------------------------------------------------------------------
# Feature Scaling
#-------------------------------------------------------------------------------

training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])
print(training_set)
print(test_set)
