library(lubridate)
library(tidyverse)

# read raw data and extract date column
train_raw <- readr::read_csv(unz('/Users/fanyang/Dropbox/uiuc/cs598/UIUC_SPL/UIUC_PSL/Project2/train.csv.zip', 'train.csv'))
train_dates <- train_raw$Date

# training data from 2010-02 to 2011-02
start_date <- ymd("2010-02-01")
end_date <- start_date %m+% months(13)

# split dataset into training / testing
train_ids <- which(train_dates >= start_date & train_dates < end_date)
train = train_raw[train_ids, ]
test = train_raw[-train_ids, ]

# create the initial training data
readr::write_csv(train, 'train_ini.csv')

# create test.csv 
# removes weekly sales
test %>% 
  select(-Weekly_Sales) %>% 
  readr::write_csv('test.csv')

# create 10-fold time-series CV
num_folds <- 10
test_dates <- train_dates[-train_ids]

# month 1 --> 2011-03, and month 20 --> 2012-10.
# Fold 1 : month 1 & month 2, Fold 2 : month 3 & month 4 ...
for (i in 1:num_folds) {
  # filter fold for dates
  start_date <- ymd("2011-03-01") %m+% months(2 * (i - 1))
  end_date <- ymd("2011-05-01") %m+% months(2 * (i - 1))
  test_fold <- test %>%
    filter(Date >= start_date & Date < end_date)
  
  # write fold to a file
  readr::write_csv(test_fold, paste0('fold_', i, '.csv'))
}