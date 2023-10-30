dataFailure <- getAllFinancialsLoop(as.numeric(failure_banklist$Cert),
                                    getAllFinancials,
                                    metrics = c("IDT1RWAJR", "ASSET","ELNLOS", "INTINC", "EINTEXP", "EQR"),
                                    IDRSSD  = FALSE)

dataActives <- getAllFinancialsLoop(as.numeric(bank_infos$FED_RSSD),
                                    getAllFinancials,
                                    metrics = c("IDT1RWAJR", "ASSET","ELNLOS", "INTINC", "EINTEXP", "EQR"))


# adding labels to active and failure dataset
dataActives$Label <- 0
dataFailure$Label <- 1

df_edited <- data.frame()

# transforming them into data table and arranging dates
set.seed(123) 
active_test_dt <- as.data.table(dataActives) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d"))

failure_test_dt <- as.data.table(dataFailure) %>%
  filter(Date >= "2008-01-01" & Date <= "2013-12-31") %>%
  mutate(quarter = as.yearqtr(Date, format = "%Y-%m-%d")) 

failure_test_dt$lag1 <- lag(failure_test_dt$Date, n = 1)
failure_test_dt$lag1 <- failure_test_dt$Date - years(1)

df_list <- list(failure_test_dt,active_test_dt) 

df_edited <- rbindlist(df_list, fill = TRUE)


df_edited$Label <- as.factor(df_edited$Label)
df_edited$Date <- as.Date(df_edited$Date)

combined_df <- df_edited %>%
  select(-CERT)

combined_df <- combined_df %>% 
  group_by(IDRSSD,Date) %>%
  mutate(TICRC = IDT1RWAJR/ASSET) %>%
  select(-IDT1RWAJR, -ASSET)

combined_df <- combined_df %>% 
  group_by(IDRSSD,Date) %>%
  mutate(PLLL = ELNLOS/INTINC) %>%
  select(-ELNLOS)

combined_df <- combined_df %>% 
  group_by(IDRSSD,Date) %>%
  mutate(TIE = EINTEXP/INTINC) %>%
  select(-EINTEXP, -INTINC)


combined_df$lag1 <- ifelse(combined_df$Label == 0, as.Date(combined_df$Date), as.Date(combined_df$lag1))

combined_df$lag1 <- as.Date(combined_df$lag1)

combined_df <- combined_df[!duplicated(combined_df$IDRSSD), ]

#Splitting data into two different subset as train and validation test set

set.seed(123)
combined_df <- combined_df[,-c(2,3,5,6)]
trainIndex <- createDataPartition(combined_df$Label, p = 0.8, list = FALSE)
train_data <- combined_df[trainIndex, ]
validation_test_data <- combined_df[-trainIndex,]

train_data <- na.omit(train_data)
validation_test_data <- na.omit(validation_test_data)


### Decision Tree Model ###

#Original decision tree model

set.seed(123)
model_dt <- train(Label ~ ., 
                  data = train_data, 
                  method = "rpart")

pred_dt_model <- predict(model_dt, validation_test_data)
confusionMatrix(pred_dt_model,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Under-sampling version
set.seed(123)
undersampling_train_data <- downSample(x = train_data,
                                       y = train_data$Label)

undersampling_train_data <- undersampling_train_data[-6]


model_dt_under <- train(Label ~ ., 
                        data = undersampling_train_data, 
                        method = "rpart")

pred_dt_model_under <- predict(model_dt_under, validation_test_data)
confusionMatrix(pred_dt_model_under,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
oversampling_train_data <- upSample(x = train_data,
                                    y = train_data$Label)

oversampling_train_data <- oversampling_train_data[-6]



model_dt_over <- train(Label ~ ., 
                        data = oversampling_train_data, 
                        method = "rpart")

pred_dt_model_over <- predict(model_dt_over, validation_test_data)
confusionMatrix(pred_dt_model_over,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_dt_weights <- train(Label ~ ., 
                  data = train_data, 
                  method = "rpart",
                  weights = ifelse(train_data$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_dt_model_weights <- predict(model_dt_weights, validation_test_data)
confusionMatrix(pred_dt_model_weights,
                validation_test_data$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data$Label <- as.numeric(train_data$Label)
new_df <- SMOTE(train_data, train_data$Label)

new_df$data <- new_df$data[,-6]

new_df$data$Label <- ifelse(new_df$data$Label == "1", "0", "1")


model_dt <- train(Label ~ ., 
                  data = new_df$data, 
                  method = "rpart")

pred_dt_model <- predict(model_dt, validation_test_data)
confusionMatrix(pred_dt_model,
                validation_test_data$Label,
                mode = "everything",
                positive="1")



### RandomForest Model ###
set.seed(123) 
train_data$Label <- ifelse(train_data$Label == "1", "0", "1")
train_data$Label <- as.factor(train_data$Label)

model_rf <- ranger(Label ~ .,
                   data = train_data) # mtry = 2


preds_rf_model <- predict(model_rf, validation_test_data)

confusionMatrix(preds_rf_model$predictions,       
                validation_test_data$Label,
                mode = "everything",
                positive = "1")

# Under-sampling version
set.seed(123)
model_under_rf <- ranger(Label ~ ., 
                        data = undersampling_train_data)

pred_model_under_rf <- predict(model_under_rf, validation_test_data)
confusionMatrix(pred_model_under_rf$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Over-sampling version
set.seed(123)
model_over_rf <- ranger(Label ~ ., 
                       data = oversampling_train_data)

pred_model_over_rf <- predict(model_over_rf, validation_test_data)
confusionMatrix(pred_model_over_rf$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Using weights parameter

set.seed(123)
model_weights_rf <- ranger(Label ~ .,
                        data = train_data,
                        case.weights = ifelse(train_data$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_rf <- predict(model_weights_rf, validation_test_data)
confusionMatrix(pred_model_weights_rf$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data$Label <- as.numeric(train_data$Label)
new_df <- SMOTE(train_data, train_data$Label)

new_df$data <- new_df$data[-6]

new_df$data$Label <- ifelse(new_df$data$Label == "1", "0", "1")
new_df$data$Label <- as.factor(new_df$data$Label)


model_smote_rf <- randomForest(Label ~ .,
                                ntree = 100,
                                data = new_df$data)


pred_model_smote_rf <- predict(model_smote_rf, validation_test_data)
confusionMatrix(pred_model_smote_rf,
                validation_test_data$Label,
                mode = "everything",
                positive = "1")


### extraTrees ###
set.seed(123)
train_data$Label <- ifelse(train_data$Label == "1", "0", "1")
train_data$Label <- as.factor(train_data$Label)


rfe <- ranger(Label ~.,
              splitrule = "extratrees",
              data = train_data)

preds_rf_model_ext <- predict(rfe, validation_test_data)

confusionMatrix(preds_rf_model_ext$predictions,       
                validation_test_data$Label,
                mode = "everything",
                positive = "1")


# Under-sampling version

set.seed(123)
model_under_ext <- ranger(Label ~ ., 
                         splitrule = "extratrees",
                         data = undersampling_train_data)

pred_model_under_ext <- predict(model_under_ext, validation_test_data)
confusionMatrix(pred_model_under_ext$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")

# Over-sampling version

set.seed(123)
model_over_ext <- ranger(Label ~ ., 
                         splitrule = "extratrees",
                        data = oversampling_train_data)

pred_model_over_ext <- predict(model_over_ext, validation_test_data)
confusionMatrix(pred_model_over_ext$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")



# Using weights parameter

set.seed(123)
model_weights_ext <- ranger(Label ~ .,
                           splitrule = "extratrees",
                           data = train_data,
                           case.weights = ifelse(train_data$Label == 1, 5, 1)) # We can change the weights of minority class. After that we can decide which one will be using.

pred_model_weights_ext <- predict(model_weights_ext, validation_test_data)
confusionMatrix(pred_model_weights_ext$predictions,
                validation_test_data$Label,
                mode = "everything",
                positive="1")


# SMOTE
set.seed(123)
train_data$Label <- as.numeric(train_data$Label)
new_df <- SMOTE(train_data, train_data$Label)

new_df$data <- new_df$data[-6]

new_df$data$Label <- ifelse(new_df$data$Label == "1", "0", "1")
new_df$data$Label <- as.factor(new_df$data$Label)


model_smote_ext <- randomForest(Label ~ .,
                                  splitrule = "extratrees",
                                  ntree = 100,
                                  data = new_df$data)


pred_model_smote_ext <- predict(model_smote_ext, validation_test_data)
confusionMatrix(pred_model_smote_rfe,
                validation_test_data$Label,
                mode = "everything",
                positive = "1")
