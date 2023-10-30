dataFailureSecondData <-
  getAllFinancialsLoop(
    as.numeric(failure_banklist$Cert),
    getAllFinancials,
    metrics = c("IDT1RWAJR", "ASSET", "ROE", "INTEXPYQ", "RBC1AAJ", "NIMY"),
    IDRSSD  = FALSE
  )

dataActivesSecondData <-
  getAllFinancialsLoop(
    as.numeric(bank_infos$FED_RSSD),
    getAllFinancials,
    metrics = c("IDT1RWAJR", "ASSET", "ROE", "INTEXPYQ", "RBC1AAJ", "NIMY")
  )


# adding labels to active and failure dataset
dataActivesSecondData$Label <- 0
dataFailureSecondData$Label <- 1

df_edited2 <- data.frame()
maxDate <- max(dataActivesSecondData$Date)

# transforming them into data table and arranging dates
set.seed(123)
second_active_test_dt <- as.data.table(dataActivesSecondData) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))

second_failure_test_dt <- as.data.table(dataFailureSecondData) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))

second_failure_test_dt$lag1 <-
  lag(second_failure_test_dt$Date, n = 1)
second_failure_test_dt$lag1 <-
  second_failure_test_dt$Date - years(1)



df_list2 <- list(second_failure_test_dt, second_active_test_dt)

df_edited2 <- rbindlist(df_list2, fill = TRUE)

df_edited2$Label <- as.factor(df_edited2$Label)
df_edited2$Date <- as.Date(df_edited2$Date)

combined_df2 <- df_edited2 %>%
  select(-CERT)

combined_df2 <- combined_df2 %>%
  group_by(IDRSSD, Date) %>%
  mutate(TICRC = IDT1RWAJR / ASSET) %>%
  select(-IDT1RWAJR,-ASSET)


combined_df2$lag1 <-
  ifelse(combined_df2$Label == 0,
         as.Date(combined_df2$Date),
         as.Date(combined_df2$lag1))

combined_df2$lag1 <- as.Date(combined_df2$lag1)

combined_df2 <- combined_df2[!duplicated(combined_df2$IDRSSD),]


#Splitting data into two different subset as train and validation test set

set.seed(123)
combined_df2 <- combined_df2[,-c(5, 6, 8, 9)]
trainIndex <-
  createDataPartition(combined_df2$Label, p = 0.8, list = FALSE)
train_data2 <- combined_df2[trainIndex,]
validation_test_data2 <- combined_df2[-trainIndex, ]

train_data2 <- na.omit(train_data2)
validation_test_data2 <- na.omit(validation_test_data2)



### Decision Tree Model ###

#Original decision tree model
set.seed(123)
model_dt2 <- train(Label ~ .,
                   data = train_data2,
                   method = "rpart")

pred_dt_model2 <- predict(model_dt2, validation_test_data2)
confusionMatrix(pred_dt_model2,
                validation_test_data2$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version
set.seed(123)
undersampling_train_data2 <-
  downSample(x = train_data2,
             y = train_data2$Label)

undersampling_train_data2 <- undersampling_train_data2[-7]


model_dt_under2 <- train(Label ~ .,
                         data = undersampling_train_data2,
                         method = "rpart")

pred_dt_model_under2 <-
  predict(model_dt_under2, validation_test_data2)
confusionMatrix(
  pred_dt_model_under2,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Over-sampling version
set.seed(123)
oversampling_train_data2 <-
  upSample(x = train_data2,
           y = train_data2$Label)

oversampling_train_data2 <- oversampling_train_data2[-7]

model_dt_over2 <- train(Label ~ .,
                        data = oversampling_train_data2,
                        method = "rpart")

pred_dt_model_over2 <-
  predict(model_dt_over2, validation_test_data2)
confusionMatrix(
  pred_dt_model_over2,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Using weights parameter

set.seed(123)
model_dt_weights2 <- train(
  Label ~ .,
  data = train_data2,
  method = "rpart",
  weights = ifelse(train_data2$Label == 1, 5, 1)
) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights2 <-
  predict(model_dt_weights2, validation_test_data2)
confusionMatrix(
  pred_dt_model_weights2,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)


# SMOTE
set.seed(123)
train_data2$Label <- as.numeric(train_data2$Label)
new_df2 <- SMOTE(train_data2, train_data2$Label)

new_df2$data <- new_df2$data[,-7]

new_df2$data$Label <- ifelse(new_df2$data$Label == "1", "0", "1")

model_dt_smote_2 <- train(Label ~ .,
                          data = new_df2$data,
                          method = "rpart")

pred_dt_model2 <- predict(model_dt_smote_2, validation_test_data2)
confusionMatrix(pred_dt_model2,
                validation_test_data2$Label,
                mode = "everything",
                positive = "1")


### RandomForest Model ###
set.seed(123)
train_data2$Label <- ifelse(train_data2$Label == "1", "0", "1")
train_data2$Label <- as.factor(train_data2$Label)



model_rf2 <- ranger(Label ~ .,
                    data = train_data2) # mtry = 2 (default)


preds_rf_model2 <- predict(model_rf2, validation_test_data2)

confusionMatrix(
  preds_rf_model2$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Under-sampling version

set.seed(123)
model_under2_rf <- ranger(Label ~ .,
                          data = undersampling_train_data2)

pred_model_under2_rf <-
  predict(model_under2_rf, validation_test_data2)
confusionMatrix(
  pred_model_under2_rf$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Over-sampling version

set.seed(123)
model_over2_rf <- ranger(Label ~ .,
                         data = oversampling_train_data2)

pred_model_over2_rf <-
  predict(model_over2_rf, validation_test_data2)
confusionMatrix(
  pred_model_over2_rf$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Using weights parameter

set.seed(123)
model_weights2_rf <- ranger(Label ~ .,
                            data = train_data2,
                            case.weights = ifelse(train_data2$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights2_rf <-
  predict(model_weights2_rf, validation_test_data2)
confusionMatrix(
  pred_model_weights2_rf$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)


# SMOTE
set.seed(123)
train_data2$Label <- as.numeric(train_data2$Label)
new_df2 <- SMOTE(train_data2, train_data2$Label)

new_df2$data <- new_df2$data[-7]

new_df2$data$Label <- ifelse(new_df2$data$Label == "1", "0", "1")
new_df2$data$Label <- as.factor(new_df2$data$Label)

model_smote_2_rf <- randomForest(Label ~ .,
                                 ntree = 100,
                                 data = new_df2$data)


pred_model2_smote_rf <- predict(model_smote_2_rf, validation_test_data2)
confusionMatrix(pred_model2_smote_rf,
                validation_test_data2$Label,
                mode = "everything",
                positive = "1")

### extraTrees ###
set.seed(123)
train_data2$Label <- ifelse(train_data2$Label == "1", "0", "1")
train_data2$Label <- as.factor(train_data2$Label)


rfe2 <- ranger(Label ~ .,
               splitrule = "extratrees",
               data = train_data2)


preds_model_ext2 <- predict(rfe2, validation_test_data2)

confusionMatrix(
  preds_model_ext2$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)


# Under-sampling version

set.seed(123)
model_under2_ext <- ranger(Label ~ .,
                           splitrule = "extratrees",
                           data = undersampling_train_data2)

pred_model_under2_ext <-
  predict(model_under2_ext, validation_test_data2)
confusionMatrix(
  pred_model_under2_ext$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Over-sampling version

set.seed(123)
model_over2_ext <- ranger(Label ~ .,
                          splitrule = "extratrees",
                          data = oversampling_train_data2)

pred_model_over2_ext <-
  predict(model_over2_ext, validation_test_data2)
confusionMatrix(
  pred_model_over2_ext$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)

# Using weights parameter

set.seed(123)
model_weights2_ext <- ranger(
  Label ~ .,
  splitrule = "extratrees",
  data = train_data2,
  case.weights = ifelse(train_data2$Label == 1, 5, 1)
) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights2_ext <-
  predict(model_weights2_ext, validation_test_data2)
confusionMatrix(
  pred_model_weights2_ext$predictions,
  validation_test_data2$Label,
  mode = "everything",
  positive = "1"
)


# SMOTE
set.seed(123)
train_data2$Label <- as.numeric(train_data2$Label)
new_df2 <- SMOTE(train_data2, train_data2$Label)

new_df2$data <- new_df2$data[-7]

new_df2$data$Label <- ifelse(new_df2$data$Label == "1", "0", "1")
new_df2$data$Label <- as.factor(new_df2$data$Label)


model_smote2_ext <- randomForest(Label ~ .,
                                  splitrule = "extratrees",
                                 ntree = 100,
                                 data = new_df2$data)


pred_model2_smote_ext <- predict(model_smote2_ext, validation_test_data2)
confusionMatrix(pred_model2_smote_ext,
                validation_test_data2$Label,
                mode = "everything",
                positive = "1")
