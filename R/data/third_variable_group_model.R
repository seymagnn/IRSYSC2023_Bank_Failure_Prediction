dataFailureThirdData <- getAllFinancialsLoop(as.numeric(failure_banklist$Cert),
                                             getAllFinancials,
                                             metrics = c("EQQR", "NIMYQ","LNATRESR", "LNLSNTQR", "NONIXAYQ", "ROAQ"),
                                             IDRSSD  = FALSE)

dataActivesThirdData <- getAllFinancialsLoop(as.numeric(bank_infos$FED_RSSD),
                                             getAllFinancials,
                                             metrics = c("EQQR", "NIMYQ","LNATRESR", "LNLSNTQR", "NONIXAYQ", "ROAQ"))

# adding labels to active and failure dataset
dataActivesThirdData$Label <- 0
dataFailureThirdData$Label <- 1


# transforming them into data table and arranging dates
set.seed(123) 
third_active_test_dt <- as.data.table(dataActivesThirdData) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))

third_failure_test_dt <- as.data.table(dataFailureThirdData) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d")) 

third_failure_test_dt$lag1 <- lag(third_failure_test_dt$Date, n = 1)
third_failure_test_dt$lag1 <- third_failure_test_dt$Date - years(1)


df_list3 <- list(third_failure_test_dt,third_active_test_dt) 

df_edited3 <- rbindlist(df_list3, fill = TRUE)


df_edited3$Label <- as.factor(df_edited3$Label)
df_edited3$Date <- as.Date(df_edited3$Date)

combined_df3 <- df_edited3 %>%
  select(-CERT)


combined_df3$lag1 <- ifelse(combined_df3$Label == 0, as.Date(combined_df3$Date), as.Date(combined_df3$lag1))

combined_df3$lag1 <- as.Date(combined_df3$lag1)

combined_df3 <- combined_df3[!duplicated(combined_df3$IDRSSD), ]


#Splitting data into two different subset as train and validation test set

set.seed(123)
combined_df3 <- combined_df3[,-c(5, 6, 8, 9)]
trainIndex <- createDataPartition(combined_df3$Label, p = 0.8, list = FALSE)
train_data3 <- combined_df3[trainIndex, ]
validation_test_data3 <- combined_df3[-trainIndex,]

train_data3 <- na.omit(train_data3)
validation_test_data3 <- na.omit(validation_test_data3)



### Decision Tree Model ###

#Original decision tree model
set.seed(123)
model_dt3 <- train(Label ~ ., 
                   data = train_data3, 
                   method = "rpart")

pred_dt_model3 <- predict(model_dt3, validation_test_data3)
confusionMatrix(pred_dt_model3,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")

# Under-sampling version
set.seed(123)
undersampling_train_data3 <- downSample(x = train_data3,
                                        y = train_data3$Label)

undersampling_train_data3 <- undersampling_train_data3[,-6]


model_dt_under3 <- train(Label ~ .,
                         data = undersampling_train_data3,
                         method = "rpart")

pred_dt_model_under3 <- predict(model_dt_under3, validation_test_data3)
confusionMatrix(pred_dt_model_under3,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
oversampling_train_data3 <- upSample(x = train_data3,
                                     y = train_data3$Label)

oversampling_train_data3 <- oversampling_train_data3[,-6]



model_dt_over3 <- train(Label ~ .,
                        data = oversampling_train_data3,
                        method = "rpart")

pred_dt_model_over3 <- predict(model_dt_over3, validation_test_data3)
confusionMatrix(pred_dt_model_over3,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_dt_weights3 <- train(Label ~ ., 
                           data = train_data3, 
                           method = "rpart",
                           weights = ifelse(train_data3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights3 <- predict(model_dt_weights3, validation_test_data3)
confusionMatrix(pred_dt_model_weights3,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data3$Label <- as.numeric(train_data3$Label)
new_df3 <- SMOTE(train_data3, train_data3$Label)

new_df3$data <- new_df3$data[,-6]

new_df3$data$Label <- ifelse(new_df3$data$Label == "1", "0", "1")



model_dt_smote_3 <- train(Label ~ ., 
                          data = new_df3$data, 
                          method = "rpart")

pred_dt_model3 <- predict(model_dt_smote_3, validation_test_data3)
confusionMatrix(pred_dt_model3,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")


### RandomForest Model ###
set.seed(123)
train_data3$Label <- ifelse(train_data3$Label == "1", "0", "1")
train_data3$Label <- as.factor(train_data3$Label)


 
model_rf3 <- ranger(Label ~ .,
                    data = train_data3) # mtry = 2 (default)


preds_rf_model3 <- predict(model_rf3, validation_test_data3)

confusionMatrix(preds_rf_model3$predictions,       
                validation_test_data3$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version

set.seed(123)
model_under3_rf <- ranger(Label ~ .,
                          data = undersampling_train_data3)

pred_model_under3_rf <-
  predict(model_under3_rf, validation_test_data3)
confusionMatrix(
  pred_model_under3_rf$predictions,
  validation_test_data3$Label,
  mode = "everything",
  positive = "1"
)

# Over-sampling version

set.seed(123)
model_over3_rf <- ranger(Label ~ .,
                         data = oversampling_train_data3)

pred_model_over3_rf <-
  predict(model_over3_rf, validation_test_data3)
confusionMatrix(
  pred_model_over3_rf$predictions,
  validation_test_data3$Label,
  mode = "everything",
  positive = "1"
)

# Using weights parameter

set.seed(123)
model_weights3_rf <- ranger(Label ~ ., 
                           data = train_data3, 
                           case.weights = ifelse(train_data3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights3_rf <- predict(model_weights3_rf, validation_test_data3)
confusionMatrix(pred_model_weights3_rf$predictions,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data3$Label <- as.numeric(train_data3$Label)
new_df3 <- SMOTE(train_data3, train_data3$Label)

new_df3$data <- new_df3$data[,-6]

new_df3$data$Label <- ifelse(new_df3$data$Label == "1", "0", "1")
new_df3$data$Label <- as.factor(new_df3$data$Label)


model_smote_3_rf <- randomForest(Label ~ .,
                                 ntree = 100,
                                 data = new_df3$data)


pred_model3_rf <- predict(model_smote_3_rf, validation_test_data3)
confusionMatrix(pred_model3_rf,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")

### extraTrees ###
set.seed(123)
train_data3$Label <- ifelse(train_data3$Label == "1", "0", "1")
train_data3$Label <- as.factor(train_data3$Label)


rfe3 <- ranger(Label ~.,
               splitrule = "extratrees",
               data = train_data3)


preds_model_ext3 <- predict(rfe3, validation_test_data3)

confusionMatrix(preds_model_ext3$predictions,       
                validation_test_data3$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version

set.seed(123)
model_under3_ext <- ranger(Label ~ .,
                           splitrule = "extratrees",
                           data = undersampling_train_data3)

pred_model_under3_ext <-
  predict(model_under3_ext, validation_test_data3)
confusionMatrix(
  pred_model_under3_ext$predictions,
  validation_test_data3$Label,
  mode = "everything",
  positive = "1"
)

# Over-sampling version

set.seed(123)
model_over3_ext <- ranger(Label ~ .,
                          splitrule = "extratrees",
                          data = oversampling_train_data3)

pred_model_over3_ext <-
  predict(model_over3_ext, validation_test_data3)
confusionMatrix(
  pred_model_over3_ext$predictions,
  validation_test_data3$Label,
  mode = "everything",
  positive = "1"
)

# Using weights parameter

set.seed(123)
model_weights3_ext <- ranger(Label ~ ., 
                            data = train_data3, 
                            splitrule = "extratrees",
                            case.weights = ifelse(train_data3$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights3_ext <- predict(model_weights3_ext, validation_test_data3)
confusionMatrix(pred_model_weights3_ext$predictions,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data3$Label <- as.numeric(train_data3$Label)
new_df3 <- SMOTE(train_data3, train_data3$Label)

new_df3$data <- new_df3$data[,-6]

new_df3$data$Label <- ifelse(new_df3$data$Label == "1", "0", "1")
new_df3$data$Label <- as.factor(new_df3$data$Label)

model_smote_3_ext <- randomForest(Label ~ .,
                                 ntree = 100,
                                 splitrule = "extratrees",
                                 data = new_df3$data)


pred_model3_ext <- predict(model_smote_3_ext, validation_test_data3)
confusionMatrix(pred_model3_ext,
                validation_test_data3$Label,
                mode = "everything",
                positive="1")








