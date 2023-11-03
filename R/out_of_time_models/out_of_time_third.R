# adding labels to active and failure dataset
dataActivesThirdData$Label <- 0
dataFailureThirdData$Label <- 1


set.seed(123) 
active_time_dt_3 <- as.data.table(dataActivesThirdData) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))


failure_time_dt_3 <- as.data.table(dataFailureThirdData) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d")) 

failure_time_dt_3$lag1 <- lag(failure_time_dt_3$Date, n = 1)
failure_time_dt_3$lag1 <- failure_time_dt_3$Date - years(1)



df_list_time_3 <- list(failure_time_dt_3,active_time_dt_3) 

df_edited_time_3 <- rbindlist(df_list_time_3, fill = TRUE)


df_edited_time_3$Label <- as.factor(df_edited_time_3$Label)
df_edited_time_3$Date <- as.Date(df_edited_time_3$Date)


combined_df_time3 <- df_edited_time_3 %>%
  select(-CERT)


combined_df_time3$lag1 <-
  ifelse(combined_df_time3$Label == 0,
         as.Date(combined_df_time3$Date),
         as.Date(combined_df_time3$lag1))

combined_df_time3$lag1 <- as.Date(combined_df_time3$lag1)

combined_df_time3<- combined_df_time3[!duplicated(combined_df_time3$IDRSSD),]


set.seed(123)
combined_df_time3 <- combined_df_time3[,-c(5,6,8,9)]
trainIndex <- createDataPartition(combined_df_time3$Label, p = 0.8, list = FALSE)
train_data_time3 <- combined_df_time3[trainIndex, ]
validation_test_data_time3 <- combined_df_time3[-trainIndex,]

train_data_time3 <- na.omit(train_data_time3)
validation_test_data_time3 <- na.omit(validation_test_data_time3)

### Decision Tree Model ###

#Original decision tree model
set.seed(123)
model_dt_time3 <- train(Label ~ ., 
                        data = train_data_time3, 
                        method = "rpart")

pred_dt_model_time3 <- predict(model_dt_time3, validation_test_data_time3)
confusionMatrix(pred_dt_model_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Under-sampling version
set.seed(123)
undersampling_train_data_time3 <- downSample(x = train_data_time3,
                                             y = train_data_time3$Label)

undersampling_train_data_time3 <- undersampling_train_data_time3[,-6]


model_dt_under_time3 <- train(Label ~ ., 
                              data = undersampling_train_data_time3, 
                              method = "rpart")

pred_dt_model_under_time3 <- predict(model_dt_under_time3, validation_test_data_time3)
confusionMatrix(pred_dt_model_under_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
oversampling_train_data_time3 <- upSample(x = train_data_time3,
                                          y = train_data_time3$Label)

oversampling_train_data_time3 <- oversampling_train_data_time3[,-6]


model_dt_over_time3 <- train(Label ~ ., 
                             data = oversampling_train_data_time3, 
                             method = "rpart")

pred_dt_model_over_time3 <- predict(model_dt_over_time3, validation_test_data_time3)
confusionMatrix(pred_dt_model_over_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_dt_weights_time3 <- train(Label ~ ., 
                                data = train_data_time3, 
                                method = "rpart",
                                weights = ifelse(train_data_time3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights_time3 <- predict(model_dt_weights_time3, validation_test_data_time3)
confusionMatrix(pred_dt_model_weights_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time3$Label <- as.numeric(train_data_time3$Label)
new_df_time3 <- SMOTE(train_data_time3, train_data_time3$Label)

new_df_time3$data <- new_df_time3$data[,-6]

new_df_time3$data$Label <- ifelse(new_df_time3$data$Label == "1", "0", "1")


model_dt_smote_time3 <- train(Label ~ ., 
                              data = new_df_time3$data, 
                              method = "rpart")

pred_dt_model_smote_time3 <- predict(model_dt_smote_time3, validation_test_data_time3)
confusionMatrix(pred_dt_model_smote_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")



### RandomForest Model ###
set.seed(123) 
train_data_time3$Label <- ifelse(train_data_time3$Label == "1", "0", "1")
train_data_time3$Label <- as.factor(train_data_time3$Label)


model_rf_time3 <- ranger(Label ~ .,
                         data = train_data_time3) # mtry = 2


preds_rf_model_time3 <- predict(model_rf_time3, validation_test_data_time3)

confusionMatrix(preds_rf_model_time3$predictions,       
                validation_test_data_time3$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version
set.seed(123)
model_under_rf_time3 <- ranger(Label ~ ., 
                               data = undersampling_train_data_time3)

pred_model_under_rf_time3 <- predict(model_under_rf_time3, validation_test_data_time3)
confusionMatrix(pred_model_under_rf_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
model_over_rf_time3 <- ranger(Label ~ ., 
                              data = oversampling_train_data_time3)

pred_model_over_rf_time3 <- predict(model_over_rf_time3, validation_test_data_time3)
confusionMatrix(pred_model_over_rf_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_weights_rf_time3 <- ranger(Label ~ .,
                                 data = train_data_time3,
                                 case.weights = ifelse(train_data_time3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_rf_time3 <- predict(model_weights_rf_time3, validation_test_data_time3)
confusionMatrix(pred_model_weights_rf_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time3$Label <- as.numeric(train_data_time3$Label)
new_df_time3 <- SMOTE(train_data_time3, train_data_time3$Label)

new_df_time3$data <- new_df_time3$data[,-6]

new_df_time3$data$Label <- ifelse(new_df_time3$data$Label == "1", "0", "1")
new_df_time3$data$Label <- as.factor(new_df_time3$data$Label)


model_smote_rf_time3 <- randomForest(Label ~ .,
                                     ntree = 100,
                                     data = new_df_time3$data)


pred_model_smote_rf_time3 <- predict(model_smote_rf_time3, validation_test_data_time3)
confusionMatrix(pred_model_smote_rf_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive = "1")


### extraTrees ###
set.seed(123)
train_data_time3$Label <- ifelse(train_data_time3$Label == "1", "0", "1")
train_data_time3$Label <- as.factor(train_data_time3$Label)


rfe_time3 <- ranger(Label ~.,
                    splitrule = "extratrees",
                    data = train_data_time3)


preds_rf_model_ext_time3 <- predict(rfe_time3, validation_test_data_time3)

confusionMatrix(preds_rf_model_ext_time3$predictions,       
                validation_test_data_time3$Label,
                mode = "everything",
                positive = "1")


# Under-sampling version

set.seed(123)
model_under_ext_time3 <- ranger(Label ~ ., 
                                splitrule = "extratrees",
                                data = undersampling_train_data_time3)

pred_model_under_ext_time3 <- predict(model_under_ext_time3, validation_test_data_time3)
confusionMatrix(pred_model_under_ext_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")

# Over-sampling version

set.seed(123)
model_over_ext_time3 <- ranger(Label ~ ., 
                               splitrule = "extratrees",
                               data = oversampling_train_data_time3)

pred_model_over_ext_time3 <- predict(model_over_ext_time3, validation_test_data_time3)
confusionMatrix(pred_model_over_ext_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")



# Using weights parameter

set.seed(123)
model_weights_ext_time3 <- ranger(Label ~ .,
                                  splitrule = "extratrees",
                                  data = train_data_time3,
                                  case.weights = ifelse(train_data_time3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_ext_time3 <- predict(model_weights_ext_time3, validation_test_data_time3)
confusionMatrix(pred_model_weights_ext_time3$predictions,
                validation_test_data_time3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time3$Label <- as.numeric(train_data_time3$Label)
new_df_time3 <- SMOTE(train_data_time3, train_data_time3$Label)

new_df_time3$data <- new_df_time3$data[,-6]

new_df_time3$data$Label <- ifelse(new_df_time3$data$Label == "1", "0", "1")
new_df_time3$data$Label <- as.factor(new_df_time3$data$Label)


model_smote_ext_time3 <- randomForest(Label ~ .,
                                      splitrule = "extratrees",
                                      ntree = 100,
                                      data = new_df_time3$data)


pred_model_smote_ext_time3 <- predict(model_smote_ext_time3, validation_test_data_time3)
confusionMatrix(pred_model_smote_ext_time3,
                validation_test_data_time3$Label,
                mode = "everything",
                positive = "1")
