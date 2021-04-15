install.packages('ISLR')
install.packages('pROC')
library(MASS)
library(ISLR)
library(pROC)
library(glmnet)

data(Caravan)

df_testing = Caravan[1:1000, ]
df_training = Caravan[1001:(dim(Caravan)[1]), ]

df_training[,'Purchase_ind'] = 1
df_training[(df_training['Purchase'] == 'No'),'Purchase_ind'] = 0

df_testing[,'Purchase_ind'] = 1
df_testing[(df_testing['Purchase'] == 'No'),'Purchase_ind'] = 0

formula_str = 'Purchase_ind ~ '
list_x = array()
for (i in colnames(Caravan)){
  if ((i != 'Purchase') & (i != 'Purchase_ind')){
    formula_str = paste(formula_str, '+', i)
    list_x = c(list_x, i)
  }
}
list_x = list_x[2:length(list_x)]

mylogit <- glm(formula_str, data=df_training,  family = "binomial")
fit_model = mylogit
get_errors = function(fit_model){
  df_testing[, 'prediction'] = predict(fit_model, newdata = df_testing, type='response')
  df_testing[df_testing[, 'prediction'] > 0.25, 'prediction_ind'] = 1
  df_testing[df_testing[, 'prediction'] <= 0.25, 'prediction_ind'] = 0
  
  # a1
  print(sum(df_testing[df_testing[, 'Purchase_ind'] == 0, 'prediction_ind'] == 1))
  # b1
  print(sum(df_testing[df_testing[, 'Purchase_ind'] == 1, 'prediction_ind'] == 0))
  
  # c1
  test_roc = roc(df_testing$Purchase_ind ~ df_testing$prediction_ind, plot = TRUE, print.auc = TRUE)
  print(as.numeric(test_roc$auc))
}

get_errors(mylogit)

# 2

step_model = stepAIC(object = glm(Purchase_ind ~ 1,
                                  data=df_training,  family = "binomial"), direction = "forward", trace = TRUE, 
                     scope = formula(formula_str) )

dim(summary(step_model)$coefficients) - 1

get_errors(step_model)


# 3 

step_model = stepAIC(object = glm(Purchase_ind ~ 1,
                                  data=df_training,  family = "binomial"), direction = "forward", trace = TRUE, 
                     scope = formula(formula_str),
                     k=log(dim(df_training)[1]))


dim(summary(step_model)$coefficients) - 1

get_errors(step_model)


# 4 

lasso_model = glmnet(y = as.matrix(df_training[, 'Purchase_ind']), 
                     x= as.matrix(df_training[, list_x]), 
                     alpha=1, 
                     family='binomial', lambda = 0.004)

lasso_model$beta
lasso_model$df

summary(lasso_model)

df_testing[, 'prediction'] = predict(lasso_model, newx = as.matrix(df_testing[, list_x]), type='response')
df_testing[df_testing[, 'prediction'] > 0.25, 'prediction_ind'] = 1
df_testing[df_testing[, 'prediction'] <= 0.25, 'prediction_ind'] = 0

# a
print(sum(df_testing[df_testing[, 'Purchase_ind'] == 0, 'prediction_ind'] == 1))
# b
print(sum(df_testing[df_testing[, 'Purchase_ind'] == 1, 'prediction_ind'] == 0))

# c
test_roc = roc(df_testing$Purchase_ind ~ df_testing$prediction_ind, plot = TRUE, print.auc = TRUE)
print(as.numeric(test_roc$auc))

