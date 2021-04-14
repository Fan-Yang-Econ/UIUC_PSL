library(MASS)
library(splines)
# attach(Boston)

lm_model = lm(nox ~ poly(dis,degree=3), data=Boston)

sum(lm_model$residuals^2)

predict(lm_model, newdata = data.frame(dis=6))

summary(lm_model)

lm_model= lm(nox ~ poly(dis,degree=4), data=Boston)
sum(lm_model$residuals^2)

lm(nox ~ bs(dis, df=3), data=Boston)
lm(nox ~ bs(dis, df=3, intercept = FALSE), data=Boston)


myfit1 = lm(nox ~ bs(dis, df=3), data=Boston)
lm(nox ~ poly(dis, 3), data=Boston)


lm(nox ~ bs(dis, df=4), data=Boston)

lm(nox ~ bs(dis, df= 4, intercept=TRUE), data=Boston)
lm(nox ~ bs(dis, df= 5, intercept=TRUE), data=Boston)
lm(nox ~ bs(dis, knots=quantile(dis, prob=c(0.25, 0.5, 0.75))), data=Boston)
   

lm(nox ~ bs(dis, knots=median(dis)), data=Boston) 
