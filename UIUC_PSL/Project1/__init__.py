
# Instruction
# https://liangfgithub.github.io/S21/Project1_S21.nb.html

# Project 1: R/Python Packages
# This post is under construction ~~~
#
# R packages:
# dplyr, tidyr, reshape2
# CARET
# randomForest, xgboost, GBM, lightGBM
# glmnet
# ggplot2
#
# Python packages:
# pandas
# scipy
# numpy
# xgboost, lightGBM
# sklearn
# matplotlib



# A report (3 pages maximum, PDF or HTML) that provides
#
# technical details ( e.g., pre-processing, implementation details if not trivial) for the models you use, and
#
# any interesting findings from your analysis.
#
# In addition, report the accuracy on the test data (see evaluation metric given below), running time of your code and the computer system you use (e.g., Macbook Pro, 2.53 GHz, 4GB memory, or AWS t2.large) for the 10 training/test splits. You do not need to submit the part of the code that evaluates test accuracy.
#
# In your report, describe the techincal details of your code, summarize your findings, and do not copy-and-paste your code to the report.





# Code Evaluation
# We will run the command “source(mymain.R)” in a directory, in which there are only two files: train.csv and test.csv, one of the 10 training/test splits.
#
# train.csv: 83 columns;
# test.csv: 82 columns without the column “Sale_Price”.
# After running your code, we should see Two txt files in the same directory named “mysubmission1.txt” and “mysubmission2.txt.” Each submission file contains prediction on the test data from a model.
#
# Submission File Format. The file should have the following format (do not forget the comma between PID and Sale_Price):
#
# PID,  Sale_Price
# 528221060,  169000.7
# 535152150,  14523.6
# 533130020,  195608.2
# Evaluation Metric. Submission are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted price and the logarithm of the observed sales price. Our evaluation R code looks like the following:
#
# # Suppose we have already read test_y.csv in as a two-column
# # data frame named "test.y":
# # col 1: PID
# # col 2: Sale_Price
#
# pred <- read.csv("mysubmission1.txt")
# names(test.y)[2] <- "True_Sale_Price"
# pred <- merge(pred, test.y, by="PID")
# sqrt(mean((log(pred$Sale_Price) - log(pred$True_Sale_Price))^2))
# Performance Target. Your performance is based on the minimal RMSE from the two models. Full credit for submissions with minimal RMSE less than
#
# 0.125 for the first 5 training/test splits and
# 0.135 for the remaining 5 training/test splits.