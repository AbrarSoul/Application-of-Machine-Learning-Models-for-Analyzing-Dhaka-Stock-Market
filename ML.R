# Load data
library(readxl)
data_final <- read_excel("ACI.xlsx")
#mydata <- mydata1[,c(2:13)]
dim(data_final)
boxplot(data_final)
summary(data_final)

# Training and test model
library(caret)
set.seed(123)
train_indices <- createDataPartition(data_final$Open, p = 0.70, list = FALSE)
train_data <- data_final[train_indices, ]
test_data <- data_final[-train_indices, ]

# Define the training control object for cross-validation
#ctrl <- trainControl(method = "cv", number = 2)
set.seed(1234)
# Define the training control with 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Define the grid of hyperparameters to search over
tune_grid <- expand.grid(mtry = c(2, 5, 10))

# Random Forest model
library(randomForest)
rf_model <- train(Open ~ ., data = train_data, method = "rf", ntree = 500, 
                  trControl = train_control, tuneGrid = tune_grid, preProcess=c('center','scale'))
rf_preds_train <- predict(rf_model, newdata = train_data)
rf_preds_test <- predict(rf_model, newdata = test_data)
print(rf_model)
cor(rf_preds_test, test_data$Open)



# Support Vector Machine model
set.seed(10)
library(e1071)
train_control <- trainControl(method = "cv", number = 5)
# Define tuning grid
svm_grid <- expand.grid(C = 10^(-2:2), sigma = c(0.01, 0.1, 1))
svm_model <- train(Open ~ ., data = train_data, method = "svmRadial", trControl = train_control, tuneGrid = svm_grid)
svm_preds_train <- predict(svm_model, newdata = train_data)
svm_preds_test <- predict(svm_model, newdata = test_data)
print(svm_model)
summary(svm_model)



# Model accuracy for training and testing
library(caret)
rf_r2_train <- caret::R2(rf_preds_train, train_data$Open)
rf_r2_test <- caret::R2(rf_preds_test, test_data$Open)
rf_mae_train <- caret::MAE(rf_preds_train, train_data$Open)
rf_mae_test <- caret::MAE(rf_preds_test, test_data$Open)
rf_rmse_train <- caret::RMSE(rf_preds_train, train_data$Open)
rf_rmse_test <- caret::RMSE(rf_preds_test, test_data$Open)



svm_r2_train <- caret::R2(svm_preds_train, train_data$Open)
svm_r2_test <- caret::R2(svm_preds_test, test_data$Open)
svm_mae_train <- caret::MAE(svm_preds_train, train_data$Open)
svm_mae_test <- caret::MAE(svm_preds_test, test_data$Open)
svm_rmse_train <- caret::RMSE(svm_preds_train, train_data$Open)
svm_rmse_test <- caret::RMSE(svm_preds_test, test_data$Open)


# Justification of accuracy in r-square, MAE, and RMSE
cat("Random Forest R2 Training: ", rf_r2_train, "\n")
cat("Random Forest R2 Testing: ", rf_r2_test, "\n")
cat("Random Forest MAE Training: ", rf_mae_train, "\n")
cat("Random Forest MAE Testing: ", rf_mae_test, "\n")
cat("Random Forest RMSE Training: ", rf_rmse_train, "\n")
cat("Random Forest RMSE Testing: ", rf_rmse_test, "\n\n")


cat("Support Vector Machine R2 Training: ", svm_r2_train, "\n")
cat("Support Vector Machine R2 Testing: ", svm_r2_test, "\n")
cat("Support Vector Machine MAE Training: ", svm_mae_train, "\n")
cat("Support Vector Machine MAE Testing: ", svm_mae_test, "\n")
cat("Support Vector Machine RMSE Training: ", svm_rmse_train, "\n")
cat("Support Vector Machine RMSE Testing: ", svm_rmse_test, "\n\n")



# Random Forest Plot
library(ggplot2)

# Create data frame for plotting
rf_plot <- data.frame(actual = test_data$Open, predicted = rf_preds_test)

# Create scatterplot with regression line
ggplot(rf_plot, aes(x = actual, y = predicted)) +
  geom_point(shape = 21, fill = "blue", size=3) +
  geom_abline(intercept = 0, slope = 1, linetype = "solid", color = "red", size= 0.8) +
  labs(title="RF Model")+
  xlab("Actual Open") +
  ylab("Predicted Open")+
  # Add text box for R2 value
  theme_bw()

ggsave("RF.tiff", height=6, width=8,dpi=300)

#SVM Plot
# Convert the data to a dataframe (Testing)
data <- data.frame(actual = test_data$Open, predicted = svm_preds_test)

# Plot the data
ggplot(data, aes(x = actual, y = predicted)) +
  geom_point(shape = 21, fill = "blue", size=3) +
  geom_abline(intercept = 0, slope = 1, linetype = "solid", color = "red", size= 0.8) +
  labs(title="SVM Model")+
  labs(x = "Actual Open", y = "Predicted Open") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme_bw()
ggsave("SVM.tiff", height=6, width=8,dpi=300)