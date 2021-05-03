install.packages('ISLR')
install.packages('pROC')

library(ISLR)
library(pROC)
library(glmnet)
library(MASS)

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
  test_roc = roc(df_testing$Purchase_ind, df_testing$prediction)
  
  print(auc(test_roc))
}

get_errors(mylogit)

# 2

step_model = stepAIC(object = glm(Purchase_ind ~ 1,
                                  data=df_training,  family = "binomial"), direction = "forward", trace = TRUE, 
                     scope = formula(formula_str) )

print(dim(summary(step_model)$coefficients)[1] - 1)

get_errors(step_model)


# 3 

step_model = stepAIC(object = glm(Purchase_ind ~ 1,
                                  data=df_training,  family = "binomial"), direction = "forward", trace = TRUE, 
                     scope = formula(formula_str),
                     k=log(dim(df_training)[1]))


print(dim(summary(step_model)$coefficients)[1] - 1)

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
test_roc = roc(df_testing$Purchase_ind, df_testing$prediction)
print(auc(test_roc))


























library(ISLR)
library(pROC)
library(MASS)
mypackages = c("MASS", "glmnet")   # required packages
tmp = setdiff(mypackages, rownames(installed.packages()))  # packages need to be installed
if (length(tmp) > 0) install.packages(tmp)
lapply(mypackages, require, character.only = TRUE)
set.seed(2134)

data(Caravan)
test =Caravan[1:1000,]
train=Caravan[1001:5822, ]
full_model = glm(Purchase~., data=train, family=binomial)
test.Y = test[, 'Purchase']
test.Y.pred = predict(full_model, newdata = test, type = 'response')
sum(test.Y == 'Yes' & test.Y.pred <0.25)
sum(test.Y == 'No' & test.Y.pred > 0.25)
roc_obj <- roc(test.Y, test.Y.pred)
auc(roc_obj)

#y_proba <- predict(full_model, test$Purchace, type = 'response')
#roc_obj auc(test.Y, y_proba)


fit1 = glm(Purchase~., data=train, family=binomial)
fit2 = glm(Purchase~ 1, data=train, family=binomial)
step.model = stepAIC(fit2, direction = "forward", scope=list(upper=fit1,lower=fit2), trace=1)
test.Y.pred = predict(step.model, newdata = test, type = 'response')
sum(test.Y == 'Yes' & test.Y.pred <0.25)
sum(test.Y == 'No' & test.Y.pred > 0.25)
roc_obj <- roc(test.Y, test.Y.pred)
auc(roc_obj)


n=dim(train)[1]
step.model2 = stepAIC(fit2, direction = "forward", scope=list(upper=fit1,lower=fit2), trace=0, k=log(n))
test.Y.pred = predict(step.model2, newdata = test, type = 'response')
sum(test.Y == 'No' & test.Y.pred > 0.25)
sum(test.Y == 'Yes' & test.Y.pred <0.25)
roc_obj <- roc(test.Y, test.Y.pred)
auc(roc_obj)



p=dim(train)[2]
X=data.matrix(train[,-p]);
Y=train[,p];
heart.l1=glmnet(X,Y,family="binomial",alpha=1, lambda = 0.004)
coef = coef(heart.l1)
sum(coef!= 0)

#coef = predict(heart.l1, lambda=0.004, type="coefficients")
#sum(coef!= 0)
test.Y.pred =predict(heart.l1, newx =data.matrix(test[,-p]), exact =TRUE, type = 'response')
sum(test.Y == 'No' & test.Y.pred > 0.25)
sum(test.Y == 'Yes' & test.Y.pred <0.25)

sum(test.Y.pred < 0.25)

roc.glmnet(heart.l1)

roc_obj <- pROC::roc(test.Y, test.Y.pred)
pROC::auc(roc_obj)
