mypackages = c("MASS", "glmnet")   # required packages
tmp = setdiff(mypackages, rownames(installed.packages()))  # packages need to be installed
if (length(tmp) > 0) install.packages(tmp)
lapply(mypackages, require, character.only = TRUE)
set.seed(2134)

myData = Boston
names(myData)[14] = "Y"
iLog = c(1, 3, 5, 6, 8, 9, 10, 14);
myData[, iLog] = log(myData[, iLog]);
myData[, 2] = myData[, 2] / 10;
myData[, 7] = myData[, 7]^2.5 / 10^4
myData[, 11] = exp(0.4 * myData[, 11]) / 1000;
myData[, 12] = myData[, 12] / 100;
myData[, 13] = sqrt(myData[, 13]);

# Move the last column of myData, the response Y, to the 1st column.
myData = data.frame(Y = myData[,14], myData[,-14]);


n = dim(myData)[1]; 
p = dim(myData)[2]-1;
X = as.matrix(myData[, -1]);  # some algorithms need the matrix/vector 
Y = myData[, 1];     


prostate = read.csv('~/Dropbox/Library/UIUC_PSL/quiz3/prostate.csv')
names(prostate)
traindata = prostate[prostate$train==TRUE,]
testdata = prostate[prostate$train==FALSE,]
dim(traindata)
dim(testdata)

mylasso = glmnet(as.matrix(traindata[,c('lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45')]), 
                 traindata[,'lpsa'], alpha = 1, lambda = c(0.1, 0.5, 0.01))

mylasso
summary(mylasso)
