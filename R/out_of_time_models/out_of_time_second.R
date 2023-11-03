# adding labels to active and failure dataset
dataActivesSecondData$Label <- 0
dataFailureSecondData$Label <- 1


set.seed(123) 
active_time_dt_2 <- as.data.table(dataActivesSecondData) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))


failure_time_dt_2 <- as.data.table(dataFailureSecondData) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d")) 

failure_time_dt_2$lag1 <- lag(failure_time_dt_2$Date, n = 1)
failure_time_dt_2$lag1 <- failure_time_dt_2$Date - years(1)



df_list_time_2 <- list(failure_time_dt_2,active_time_dt_2) 

df_edited_time_2 <- rbindlist(df_list_time_2, fill = TRUE)


df_edited_time_2$Label <- as.factor(df_edited_time_2$Label)
df_edited_time_2$Date <- as.Date(df_edited_time_2$Date)


combined_df_time2 <- df_edited_time_2 %>%
  select(-CERT)

combined_df_time2 <- combined_df_time2 %>%
  group_by(IDRSSD, Date) %>%
  mutate(TICRC = IDT1RWAJR / ASSET) %>%
  select(-IDT1RWAJR,-ASSET)


combined_df_time2$lag1 <-
  ifelse(combined_df_time2$Label == 0,
         as.Date(combined_df_time2$Date),
         as.Date(combined_df_time2$lag1))

combined_df_time2$lag1 <- as.Date(combined_df_time2$lag1)

combined_df_time2 <- combined_df_time2[!duplicated(combined_df_time2$IDRSSD),]


set.seed(123)
combined_df_time2 <- combined_df_time2[,-c(5,6,8,9)]
trainIndex <- createDataPartition(combined_df_time2$Label, p = 0.8, list = FALSE)
train_data_time2 <- combined_df_time2[trainIndex, ]
validation_test_data_time2 <- combined_df_time2[-trainIndex,]

train_data_time2 <- na.omit(train_data_time2)
validation_test_data_time2 <- na.omit(validation_test_data_time2)

### Decision Tree Model ###

#Original decision tree model
set.seed(123)
model_dt_time2 <- train(Label ~ ., 
                       data = train_data_time2, 
                       method = "rpart")

pred_dt_model_time2 <- predict(model_dt_time2, validation_test_data_time2)
confusionMatrix(pred_dt_model_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Under-sampling version
set.seed(123)
undersampling_train_data_time2 <- downSample(x = train_data_time2,
                                            y = train_data_time2$Label)

undersampling_train_data_time2 <- undersampling_train_data_time2[,-7]

model_dt_under_time2 <- train(Label ~ ., 
                             data = undersampling_train_data_time2, 
                             method = "rpart")

pred_dt_model_under_time2 <- predict(model_dt_under_time2, validation_test_data_time2)
confusionMatrix(pred_dt_model_under_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
oversampling_train_data_time2 <- upSample(x = train_data_time2,
                                         y = train_data_time2$Label)

oversampling_train_data_time2 <- oversampling_train_data_time2[,-7]

model_dt_over_time2 <- train(Label ~ ., 
                            data = oversampling_train_data_time2, 
                            method = "rpart")

pred_dt_model_over_time2 <- predict(model_dt_over_time2, validation_test_data_time2)
confusionMatrix(pred_dt_model_over_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_dt_weights_time2 <- train(Label ~ ., 
                               data = train_data_time2, 
                               method = "rpart",
                               weights = ifelse(train_data_time2$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights_time2 <- predict(model_dt_weights_time2, validation_test_data_time2)
confusionMatrix(pred_dt_model_weights_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time2$Label <- as.numeric(train_data_time2$Label)
new_df_time2 <- SMOTE(train_data_time2, train_data_time2$Label)

new_df_time2$data <- new_df_time2$data[,-7]

new_df_time2$data$Label <- ifelse(new_df_time2$data$Label == "1", "0", "1")

model_dt_smote_time2 <- train(Label ~ ., 
                             data = new_df_time2$data, 
                             method = "rpart")

pred_dt_model_smote_time2 <- predict(model_dt_smote_time2, validation_test_data_time2)
confusionMatrix(pred_dt_model_smote_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")



### RandomForest Model ###
set.seed(123) 
train_data_time2$Label <- ifelse(train_data_time2$Label == "1", "0", "1")
train_data_time2$Label <- as.factor(train_data_time2$Label)


model_rf_time2 <- ranger(Label ~ .,
                        data = train_data_time2) # mtry = 2


preds_rf_model_time2 <- predict(model_rf_time2, validation_test_data_time2)

confusionMatrix(preds_rf_model_time2$predictions,       
                validation_test_data_time2$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version
set.seed(123)
model_under_rf_time2 <- ranger(Label ~ ., 
                              data = undersampling_train_data_time2)

pred_model_under_rf_time2 <- predict(model_under_rf_time2, validation_test_data_time2)
confusionMatrix(pred_model_under_rf_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
model_over_rf_time2 <- ranger(Label ~ ., 
                             data = oversampling_train_data_time2)

pred_model_over_rf_time2 <- predict(model_over_rf_time2, validation_test_data_time2)
confusionMatrix(pred_model_over_rf_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_weights_rf_time2 <- ranger(Label ~ .,
                                data = train_data_time2,
                                case.weights = ifelse(train_data_time2$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_rf_time2 <- predict(model_weights_rf_time2, validation_test_data_time2)
confusionMatrix(pred_model_weights_rf_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time2$Label <- as.numeric(train_data_time2$Label)
new_df_time2 <- SMOTE(train_data_time2, train_data_time2$Label)

new_df_time2$data <- new_df_time2$data[,-7]

new_df_time2$data$Label <- ifelse(new_df_time2$data$Label == "1", "0", "1")
new_df_time2$data$Label <- as.factor(new_df_time2$data$Label)

model_smote_rf_time2 <- randomForest(Label ~ .,
                                    ntree = 100,
                                    data = new_df_time2$data)


pred_model_smote_rf_time2 <- predict(model_smote_rf_time2, validation_test_data_time2)
confusionMatrix(pred_model_smote_rf_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive = "1")


### extraTrees ###
set.seed(123)
train_data_time2$Label <- ifelse(train_data_time2$Label == "1", "0", "1")
train_data_time2$Label <- as.factor(train_data_time2$Label)


rfe_time2 <- ranger(Label ~.,
                   splitrule = "extratrees",
                   data = train_data_time2)


preds_rf_model_ext_time2 <- predict(rfe_time2, validation_test_data_time2)

confusionMatrix(preds_rf_model_ext_time2$predictions,       
                validation_test_data_time2$Label,
                mode = "everything",
                positive = "1")


# Under-sampling version

set.seed(123)
model_under_ext_time2 <- ranger(Label ~ ., 
                               splitrule = "extratrees",
                               data = undersampling_train_data_time2)

pred_model_under_ext_time2 <- predict(model_under_ext_time2, validation_test_data_time2)
confusionMatrix(pred_model_under_ext_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")

# Over-sampling version

set.seed(123)
model_over_ext_time2 <- ranger(Label ~ ., 
                              splitrule = "extratrees",
                              data = oversampling_train_data_time2)

pred_model_over_ext_time2 <- predict(model_over_ext_time2, validation_test_data_time2)
confusionMatrix(pred_model_over_ext_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")



# Using weights parameter

set.seed(123)
model_weights_ext_time2 <- ranger(Label ~ .,
                                 splitrule = "extratrees",
                                 data = train_data_time2,
                                 case.weights = ifelse(train_data_time2$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_ext_time2 <- predict(model_weights_ext_time2, validation_test_data_time2)
confusionMatrix(pred_model_weights_ext_time2$predictions,
                validation_test_data_time2$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time2$Label <- as.numeric(train_data_time2$Label)
new_df_time2 <- SMOTE(train_data_time2, train_data_time2$Label)

new_df_time2$data <- new_df_time2$data[,-7]

new_df_time2$data$Label <- ifelse(new_df_time2$data$Label == "1", "0", "1")
new_df_time2$data$Label <- as.factor(new_df_time2$data$Label)


model_smote_ext_time2 <- randomForest(Label ~ .,
                                     splitrule = "extratrees",
                                     ntree = 100,
                                     data = new_df_time2$data)


pred_model_smote_ext_time2 <- predict(model_smote_ext_time2, validation_test_data_time2)
confusionMatrix(pred_model_smote_ext_time2,
                validation_test_data_time2$Label,
                mode = "everything",
                positive = "1")
