library(ggplot2)
library(class)
set.seed(3539)


# coding 2: https://liangfgithub.github.io/S21/Coding2_SampleCode_S21.nb.html

CSIZE = 10;       # number of centers
P = 2;      

s = 1;      # sd for generating the centers within each class                    
m1 = matrix(rnorm(CSIZE*P), CSIZE, P)*s + cbind( rep(1,CSIZE), rep(0,CSIZE));
m0 = matrix(rnorm(CSIZE*P), CSIZE, P)*s + cbind( rep(0,CSIZE), rep(1,CSIZE));


generate_data = function(n=100, testing=FALSE){
  # Randomly allocate the n samples for class 1  to the 10 clusters
  id1 = sample(1:CSIZE, n, replace = TRUE);
  # Randomly allocate the n samples for class 1 to the 10 clusters
  id0 = sample(1:CSIZE, n, replace = TRUE);  
  
  s= sqrt(1/5);                               # sd for generating x. 
  
  traindata = matrix(rnorm(2*n*P), 2*n, P)*s + rbind(m1[id1,], m0[id0,])
  df_train = data.frame(traindata)
  colnames(df_train) = c('X1', 'X2')
  df_train[,'y_true_value'] = c(rep(1,n), rep(0,n))
  return (df_train)
}

cal_error_rate = function(true_value, estimated_value){
  df_ = data.frame(true_value=true_value, estimated_value=estimated_value)
  return(sum(df_['true_value'] != df_['estimated_value']) / length(true_value))
}

# LM model

quadratic_formula=y_true_value ~ X1 + X2 + I(X1^2) + I(X2^2) + I(X1*X2)

get_lm_model_error_rate = function(df_train, df_test, formula=y_true_value ~ X1 + X2){
  lm_model = lm(formula = formula, data = df_train)
  
  y_result = predict(lm_model, newdata = df_test)
  y_result[y_result > 0.5] = 1
  y_result[y_result < 0.5] = 0
  df_test['estimated_value'] = y_result
  
  true_value = df_test[,'y_true_value']
  estimated_value = df_test[,'estimated_value']
  
  return(cal_error_rate(true_value, estimated_value))
  
}




mixnorm = function(x, centers0, centers1, s){
  ## return the density ratio for a point x, where each 
  ## density is a mixture of normal with multiple components
  d1 = sum(exp(-apply((t(centers1) - x)^2, 2, sum) / (2 * s^2)))
  d0 = sum(exp(-apply((t(centers0) - x)^2, 2, sum) / (2 * s^2)))
  return (d1 / d0)
}

get_bayes_error = function(df_test, centers1=c(1,0), centers0=c(0,1)){
  
  if (is.null(centers1)){
    
    centers1 = c(
      mean(df_test[df_test['y_true_value'] == 1, 'X1']),
      mean(df_test[df_test['y_true_value'] == 1, 'X2'])
    )
  }
  
  if (is.null(centers0)){
    centers0 = c(
      mean(df_test[df_test['y_true_value'] == 0, 'X1']),
      mean(df_test[df_test['y_true_value'] == 0, 'X2'])
    )
  }
  
  list_bayes_rule = c()
  for (i in seq(1, dim(df_test)[1])){
    list_bayes_rule = append(list_bayes_rule, 
                             mixnorm(c(df_test[i, 'X1'], df_test[i, 'X2']), 
                                     centers1=centers1, 
                                     centers0=centers0,
                                     s=s)
                             )
  }
  
  list_bayes_rule[list_bayes_rule >= 1] = 1
  list_bayes_rule[list_bayes_rule < 1] = 0
  
  return(cal_error_rate(df_test[, 'y_true_value'], list_bayes_rule))
  
}

df_error_wide = data.frame()
list_k_chosen = c()
for (i in seq(1,20)){
  
  df_train=generate_data(100)
  df_test = generate_data(5000)
  
  
  error_bayes = c(
    train_error = get_bayes_error(df_train, centers1=c(1,0), centers0=c(0,1)),
    test_error = get_bayes_error(df_test, centers1=NULL, centers0=NULL),
    mode = 'Bayes'
  )
  
  error_linear = c(
    train_error = get_lm_model_error_rate(df_train, df_train),
    test_error = get_lm_model_error_rate(df_train, df_test),
    mode = 'Linear'
  )
  
  error_quadratic = c(
    train_error = get_lm_model_error_rate(df_train, df_train, formula=quadratic_formula),
    test_error = get_lm_model_error_rate(df_train, df_test, formula=quadratic_formula),
    mode = 'Quadratic'
  )
  
  K_chosen = chose_K_for_KNN(df_train)
  
  error_KNN = c(
    train_error = get_knn_error(df_train, df_train, K_chosen),
    test_error = get_knn_error(df_train, df_test, K_chosen),
    mode = 'KNN'
  )
  list_k_chosen = append(list_k_chosen, K_chosen)
  
  df_ = rbind(
    error_bayes,
    error_linear,
    error_quadratic,
    error_KNN
  )
  
  df_error_wide = rbind(df_error_wide, df_)
}


df_error_wide[,'train_error'] = as.numeric(df_error_wide[,'train_error'])
df_error_wide[,'test_error'] = as.numeric(df_error_wide[,'test_error'])

df_test_error = df_error_wide[,c('test_error', 'mode')]
df_train_error = df_error_wide[,c('train_error', 'mode')]

df_test_error[, 'error_type'] = 'test_error'
df_train_error[, 'error_type'] = 'train_error'

colnames(df_test_error)[1] = 'error'
colnames(df_train_error)[1] = 'error'

df_error = rbind(df_train_error, df_test_error)

ggplot(df_error) + geom_boxplot(mapping = aes(y = error, x=mode, color=error_type))








get_bayes_error(df_test=generate_data())
df_test=generate_data()
get_bayes_error(df_test)
get_bayes_error(df_test, centers1=NULL, centers0=NULL)

get_bayes_error(df_test=generate_data())




range(0, dim(df_train)[1])


df_train = generate_data()
df_test = generate_data(n=5000)

get_lm_model_error_rate(df_train, df_test)
get_lm_model_error_rate(df_train, df_test, formula=y_true_value ~ X1 + X2 + I(X1^2) + I(X2^2) + I(X1*X2))

get_knn_error(df_train, df_test)






# ggplot(df_train) + geom_point(mapping=aes(X1, y=X2, color=factor(y_true_value))) + 
#   geom_abline(mapping=aes(slope=-1 * lm_model$coefficients['X1'] / lm_model$coefficients['X2'],
#                       intercept=-1 * (lm_model$coefficients['(Intercept)'] - 0.5)
#                       ))


qu_model = lm(formula = y_true_value ~ X1 + X2 + I(X1^2) + I(X2^2) + I(X1*X2), data = df_train)
qu_model$coefficients
predicted_y = predict(qu_model, newdata = estimated_results)

df_sep_line = estimated_results[(predicted_y > 0.45) & (predicted_y<0.55),]


# Constructing delta
delta<-function(a,b,c){
  b^2-4*a*c
}
# Constructing Quadratic Formula
solve_qudratic <- function(a,b,c){
  if(delta(a,b,c) > 0){ # first case D>0
    x_1 = (-b+sqrt(delta(a,b,c)))/(2*a)
    x_2 = (-b-sqrt(delta(a,b,c)))/(2*a)
    result = c(x_1,x_2)
  }
  else if(delta(a,b,c) == 0){ # second case D=0
    x = -b/(2*a)
  }
}

list_x_simu = seq(min(df_train$X1), max(df_train$X1), 0.1)
predicted_sep_line <- matrix(ncol=2, nrow=length(list_x_simu))
count = 1

for (x1 in list_x_simu){
  
  c = qu_model$coefficients['(Intercept)'] - 0.5 + qu_model$coefficients['X1'] * x1 + qu_model$coefficients['I(X2^2)'] * x1^2
  b = qu_model$coefficients['X2']+ qu_model$coefficients['I(X1 * X2)'] * x1
  a = qu_model$coefficients['I(X2^2)']
  
  root_list = solve_qudratic(a, b, c)
  if (is.null(root_list)){
    next
  }
  
  X2=max(solve_qudratic(a, b, c))
         
  predicted_sep_line[count, ] = c(X1=x1, X2=X2)
  count = count + 1
}

predicted_sep_line = data.frame(predicted_sep_line)
ggplot(df_train) + geom_point(mapping=aes(X1, y=X2, color=factor(y_true_value))) +
  geom_point(mapping=aes(X1, y=X2),  data=predicted_sep_line)



estimated_results = matrix(NA, 1, 2)

for (x1 in seq(min(df_train$X1), max(df_train$X1), 0.1)){
  for (x2 in seq(min(df_train$X2), max(df_train$X2), 0.1)){
    estimated_results = rbind(c(X1=x1, X2=x2), estimated_results)
  }
}
estimated_results = data.frame(estimated_results)
estimated_results = estimated_results[!is.na(estimated_results[, 'X1']), ]

predicted_value = knn(df_train[, c('X1', 'X2')], estimated_results, df_train[, 'y_true_value'], K_chosen)

estimated_results['predicted_value'] = predicted_value
ggplot(df_train) + geom_point(mapping=aes(X1, y=X2, color=factor(y_true_value))) +
  geom_point(mapping=aes(X1, y=X2, color = factor(predicted_value)), size=0.01, data=estimated_results)




ggplot(df_train) + geom_point(mapping=aes(X1, y=X2, color=factor(y_true_value))) +
  geom_point(mapping=aes(X1, y=X2, color = factor(predicted_value)), size=0.01, data=estimated_results)



mixnorm(c(0.102019971, 0.861607534), c(0,1), c(1,0), s)
mixnorm(c(1.290606440, -0.976565742), c(0,1), c(1,0), s)



                              