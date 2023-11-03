# adding labels to active and failure dataset
dataActives$Label <- 0
dataFailure$Label <- 1


set.seed(123) 
active_time_dt <- as.data.table(dataActives) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))


failure_time_dt <- as.data.table(dataFailure) %>%
  filter(Date >= "2014-01-01" & Date <= "2023-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d")) 

failure_time_dt$lag1 <- lag(failure_time_dt$Date, n = 1)
failure_time_dt$lag1 <- failure_time_dt$Date - years(1)



df_list_time <- list(failure_time_dt,active_time_dt) 

df_edited_time <- rbindlist(df_list_time, fill = TRUE)


df_edited_time$Label <- as.factor(df_edited_time$Label)
df_edited_time$Date <- as.Date(df_edited_time$Date)


combined_df_time <- df_edited_time %>%
  select(-CERT)

combined_df_time <- combined_df_time %>% 
  group_by(IDRSSD,Date) %>%
  mutate(TICRC = IDT1RWAJR/ASSET) %>%
  select(-IDT1RWAJR, -ASSET)

combined_df_time <- combined_df_time %>% 
  group_by(IDRSSD,Date) %>%
  mutate(PLLL = ELNLOS/INTINC) %>%
  select(-ELNLOS)

combined_df_time <- combined_df_time %>% 
  group_by(IDRSSD,Date) %>%
  mutate(TIE = EINTEXP/INTINC) %>%
  select(-EINTEXP, -INTINC)


combined_df_time$lag1 <- ifelse(combined_df_time$Label == 0, as.Date(combined_df_time$Date), as.Date(combined_df_time$lag1))

combined_df_time$lag1 <- as.Date(combined_df_time$lag1)

combined_df_time <- combined_df_time[!duplicated(combined_df_time$IDRSSD), ]


set.seed(123)
combined_df_time <- combined_df_time[,-c(2,3,5,6)]
trainIndex <- createDataPartition(combined_df_time$Label, p = 0.8, list = FALSE)
train_data_time <- combined_df_time[trainIndex, ]
validation_test_data_time <- combined_df_time[-trainIndex,]

train_data_time <- na.omit(train_data_time)
validation_test_data_time <- na.omit(validation_test_data_time)

### Decision Tree Model ###

#Original decision tree model
set.seed(123)
model_dt_time <- train(Label ~ ., 
                  data = train_data_time, 
                  method = "rpart")

pred_dt_model_time <- predict(model_dt_time, validation_test_data_time)
confusionMatrix(pred_dt_model_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Under-sampling version
set.seed(123)
undersampling_train_data_time <- downSample(x = train_data_time,
                                       y = train_data_time$Label)

undersampling_train_data_time <- undersampling_train_data_time[-6]


model_dt_under_time <- train(Label ~ ., 
                        data = undersampling_train_data_time, 
                        method = "rpart")

pred_dt_model_under_time <- predict(model_dt_under_time, validation_test_data_time)
confusionMatrix(pred_dt_model_under_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
oversampling_train_data_time <- upSample(x = train_data_time,
                                    y = train_data_time$Label)

oversampling_train_data_time <- oversampling_train_data_time[-6]



model_dt_over_time <- train(Label ~ ., 
                       data = oversampling_train_data_time, 
                       method = "rpart")

pred_dt_model_over_time <- predict(model_dt_over_time, validation_test_data_time)
confusionMatrix(pred_dt_model_over_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_dt_weights_time <- train(Label ~ ., 
                          data = train_data_time, 
                          method = "rpart",
                          weights = ifelse(train_data_time$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights_time <- predict(model_dt_weights_time, validation_test_data_time)
confusionMatrix(pred_dt_model_weights_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time$Label <- as.numeric(train_data_time$Label)
new_df_time <- SMOTE(train_data_time, train_data_time$Label)

new_df_time$data <- new_df_time$data[,-6]

new_df_time$data$Label <- ifelse(new_df_time$data$Label == "1", "0", "1")


model_dt_smote_time <- train(Label ~ ., 
                  data = new_df_time$data, 
                  method = "rpart")

pred_dt_model_smote_time <- predict(model_dt_smote_time, validation_test_data_time)
confusionMatrix(pred_dt_model_smote_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")



### RandomForest Model ###
set.seed(123) 
train_data_time$Label <- ifelse(train_data_time$Label == "1", "0", "1")
train_data_time$Label <- as.factor(train_data_time$Label)

model_rf_time <- ranger(Label ~ .,
                   data = train_data_time) # mtry = 2


preds_rf_model_time <- predict(model_rf_time, validation_test_data_time)

confusionMatrix(preds_rf_model_time$predictions,       
                validation_test_data_time$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version
set.seed(123)
model_under_rf_time <- ranger(Label ~ ., 
                         data = undersampling_train_data_time)

pred_model_under_rf_time <- predict(model_under_rf_time, validation_test_data_time)
confusionMatrix(pred_model_under_rf_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
model_over_rf_time <- ranger(Label ~ ., 
                        data = oversampling_train_data_time)

pred_model_over_rf_time <- predict(model_over_rf_time, validation_test_data_time)
confusionMatrix(pred_model_over_rf_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_weights_rf_time <- ranger(Label ~ .,
                           data = train_data_time,
                           case.weights = ifelse(train_data_time$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_rf_time <- predict(model_weights_rf_time, validation_test_data_time)
confusionMatrix(pred_model_weights_rf_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time$Label <- as.numeric(train_data_time$Label)
new_df_time <- SMOTE(train_data_time, train_data_time$Label)

new_df_time$data <- new_df_time$data[-6]

new_df_time$data$Label <- ifelse(new_df_time$data$Label == "1", "0", "1")
new_df_time$data$Label <- as.factor(new_df_time$data$Label)


model_smote_rf_time <- randomForest(Label ~ .,
                               ntree = 100,
                               data = new_df_time$data)


pred_model_smote_rf_time <- predict(model_smote_rf_time, validation_test_data_time)
confusionMatrix(pred_model_smote_rf_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive = "1")


### extraTrees ###
set.seed(123)
train_data_time$Label <- ifelse(train_data_time$Label == "1", "0", "1")
train_data_time$Label <- as.factor(train_data_time$Label)


rfe_time <- ranger(Label ~.,
              splitrule = "extratrees",
              data = train_data_time)


preds_rf_model_ext_time <- predict(rfe_time, validation_test_data_time)

confusionMatrix(preds_rf_model_ext_time$predictions,       
                validation_test_data_time$Label,
                mode = "everything",
                positive = "1")


# Under-sampling version

set.seed(123)
model_under_ext_time <- ranger(Label ~ ., 
                          splitrule = "extratrees",
                          data = undersampling_train_data_time)

pred_model_under_ext_time <- predict(model_under_ext_time, validation_test_data_time)
confusionMatrix(pred_model_under_ext_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")

# Over-sampling version

set.seed(123)
model_over_ext_time <- ranger(Label ~ ., 
                         splitrule = "extratrees",
                         data = oversampling_train_data)

pred_model_over_ext_time <- predict(model_over_ext_time, validation_test_data_time)
confusionMatrix(pred_model_over_ext_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")



# Using weights parameter

set.seed(123)
model_weights_ext_time <- ranger(Label ~ .,
                            splitrule = "extratrees",
                            data = train_data_time,
                            case.weights = ifelse(train_data_time$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_ext_time <- predict(model_weights_ext_time, validation_test_data_time)
confusionMatrix(pred_model_weights_ext_time$predictions,
                validation_test_data_time$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data_time$Label <- as.numeric(train_data_time$Label)
new_df_time <- SMOTE(train_data_time, train_data_time$Label)

new_df_time$data <- new_df_time$data[,-6]

new_df_time$data$Label <- ifelse(new_df_time$data$Label == "1", "0", "1")
new_df_time$data$Label <- as.factor(new_df_time$data$Label)

model_smote_ext_time <- randomForest(Label ~ .,
                                splitrule = "extratrees",
                                ntree = 100,
                                data = new_df_time$data)


pred_model_smote_ext_time <- predict(model_smote_ext_time, validation_test_data_time)
confusionMatrix(pred_model_smote_ext_time,
                validation_test_data_time$Label,
                mode = "everything",
                positive = "1")
