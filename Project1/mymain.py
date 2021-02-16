# Step 0: Load necessary libraries
###########################################
# Step 0: Load necessary libraries
#

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(
    description="run_project_1.py")

parser.add_argument('--folder',
                    help='folder for data',
                    default='.')

FOLDER = '/Users/yafa/Dropbox/Library/UIUC_PSL/Project1/'
# parsed_args = parser.parse_args(['--folder', '/Users/yafa/Dropbox/Library/UIUC_PSL/Project1/'])

parsed_args = parser.parse_args()
FOLDER = parsed_args.folder

print(parsed_args)
###########################################
# Step 1: Preprocess training data
#         and fit two models
#
df_train = pd.read_csv(os.path.join(FOLDER, "train.csv"))




###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#
df_test = pd.read_csv(os.path.join(FOLDER, "test.csv"))
#
# YOUR CODE
#