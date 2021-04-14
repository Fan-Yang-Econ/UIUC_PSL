library(MASS)

load('/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/quiz_7/zip.train.Rdata')
load('/Users/yafa/Dropbox/Library/UIUC_PSL/UIUC_PSL/quiz_7/zip.test.Rdata')

zip.test
zip.test
zip.train

df_train = data.frame(zip.train)
formula_str = 'X1 ~ '

for (x in c(2:257)){
  formula_str = paste(formula_str, '+X', x, sep='')
}

model = lda(formula(formula_str), df_train)
model


x_predict = predict(model, newdata = data.frame(zip.test))$x
predicted_class = predict(model, newdata = data.frame(zip.test))$class



# In zip.test, there are [A] (an integer) images for digit "4", and among them, [B] (an integer) are correctly classified.

sum(zip.test[, 1] == 4)

df2 = data.frame(label_true = zip.test[, 1], predicted_label= predicted_class)
sum((df2['label_true'] ==4) & df2['predicted_label'] ==4)

# There are [C] (an integer) images in zip.test that are classified (by LDA) as digit "4", and among them, [D] (an integer) images are mis-classifed.

sum(df2['predicted_label'] ==4)
sum((df2['label_true'] != 4) & df2['predicted_label'] ==4)

df2[4, ]

dim(x_predict)
dim(zip.test)


x_predict[4, ]
